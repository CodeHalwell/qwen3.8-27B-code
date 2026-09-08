"""Teacher policies: a larger open model driving the same six-tool harness.

Distillation in this project is sequence-level and execution-verified. The
teacher acts through the deployment tool schema and the repository harness,
every attempt is graded from outside the workspace, and only verified
attempts become student data. That is the "Regenerable" lane of
docs/data-strategy.md applied to a model instead of a public trace set: the
task is kept, the actions are generated through the target adapter, and
nothing is translated or fabricated. docs/distillation.md covers what this
does and does not buy compared with logit-level distillation.

The adapter speaks the OpenAI-compatible chat-completions protocol that vLLM,
llama.cpp's server, Hugging Face Inference Providers and the vendor APIs for
the Qwen, Kimi and GLM families expose. It uses only the standard library so
the package keeps no runtime dependencies, and its transport is injectable,
which is how the tests drive it without a network.

Two conversions keep the episode loop unchanged. Outgoing, the loop's
messages (developer, user, assistant with ``reasoning_content`` and native
tool calls, tool observations) are folded into the API's shape, with tool
call ids threaded through. Incoming, the API's structured response is
rendered back into Qwen3.8's XML tool-call text, so the loop parses, stores
and grades a teacher turn exactly as it does a student turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Callable
import urllib.error
import urllib.request

from .episodes import Policy, TurnResult, answer_text
from .parsing import split_reasoning
from .schema import TOOLS, canonical_to_qwen

Transport = Callable[[dict], dict]

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class TeacherError(RuntimeError):
    """The teacher endpoint could not produce a usable turn.

    Raised inside the policy, so the episode loop records ``policy_error``:
    an infrastructure failure that is never scored against a model.
    """


class ReasoningHidden(TeacherError):
    """The endpoint returned a turn without its reasoning.

    Training the student on a think block the teacher did not actually write
    (an empty one) would teach it to skip thinking at the labelled effort.
    Rows without visible reasoning are refused at the source instead.
    """


@dataclass(frozen=True)
class TeacherPreset:
    base_url: str
    api_key_env: str | None
    note: str


# Endpoints that speak the protocol. Vendor endpoints move; the ones listed
# were read from the vendors' own documentation on the date in
# docs/references.md, and --base-url overrides any of them.
PRESETS: dict[str, TeacherPreset] = {
    "hf-inference-providers": TeacherPreset(
        "https://router.huggingface.co/v1",
        "HF_TOKEN",
        "Hugging Face router; model ids are Hub ids, optionally with a :provider suffix",
    ),
    "vllm-local": TeacherPreset("http://localhost:8000/v1", None, "vLLM OpenAI-compatible server"),
    "llama-cpp-local": TeacherPreset("http://localhost:8080/v1", None, "llama.cpp server with --jinja"),
    "moonshot": TeacherPreset(
        "https://api.moonshot.ai/v1",
        "MOONSHOT_API_KEY",
        "Kimi vendor API (K3, K2.6); thinking models return reasoning_content",
    ),
    "zai": TeacherPreset(
        "https://api.z.ai/api/paas/v4",
        "ZAI_API_KEY",
        "GLM vendor API (5.3, 5.3 Flash); enable thinking through extra_body per its docs",
    ),
}


@dataclass(frozen=True)
class TeacherConfig:
    """Everything the adapter needs to call one teacher."""

    model: str
    base_url: str
    api_key_env: str | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout_seconds: float = 300.0
    max_retries: int = 3
    # Refuse turns whose reasoning the endpoint did not expose.
    require_reasoning: bool = True
    # Interleaved-thinking APIs (Kimi K2/K3 style) need earlier reasoning
    # passed back on multi-turn tool use; vLLM accepts it; strict OpenAI-style
    # validators reject unknown message fields, so it is switchable.
    send_reasoning_back: bool = True
    # Sent as the OpenAI ``reasoning_effort`` parameter when set. Qwen3.8
    # servers honour it; others may reject or ignore it.
    reasoning_effort: str | None = None
    # Merged into the request body verbatim, for vendor-specific switches
    # such as GLM's thinking flag or vLLM's chat_template_kwargs.
    extra_body: dict = field(default_factory=dict)
    tool_choice: str = "auto"

    @property
    def label(self) -> str:
        return f"teacher:{self.model}"

    @classmethod
    def from_preset(cls, preset: str, model: str, **overrides) -> "TeacherConfig":
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; known: {', '.join(sorted(PRESETS))}")
        chosen = PRESETS[preset]
        return cls(model=model, base_url=chosen.base_url, api_key_env=chosen.api_key_env, **overrides)


# ---------------------------------------------------------------------------
# Outgoing: episode messages -> API messages


def to_api_messages(messages: list[dict], send_reasoning_back: bool = True) -> list[dict]:
    """Fold the loop's messages into chat-completions messages with call ids."""
    converted = []
    pending_ids: list[str] = []
    call_counter = 0
    for message in canonical_to_qwen(messages):
        role = message["role"]
        if role == "assistant":
            entry: dict = {"role": "assistant", "content": message.get("content") or None}
            reasoning = message.get("reasoning_content")
            if send_reasoning_back and reasoning:
                entry["reasoning_content"] = reasoning
            calls = []
            for call in message.get("tool_calls") or []:
                call_counter += 1
                call_id = f"call_{call_counter:04d}"
                pending_ids.append(call_id)
                function = call["function"]
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": json.dumps(function.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                )
            if calls:
                entry["tool_calls"] = calls
            converted.append(entry)
        elif role == "tool":
            if not pending_ids:
                raise TeacherError("tool observation without a preceding tool call")
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_ids.pop(0),
                    "name": message.get("name", ""),
                    "content": message.get("content", ""),
                }
            )
        else:
            converted.append({"role": role, "content": message.get("content", "")})
    return converted


# ---------------------------------------------------------------------------
# Incoming: API response -> native turn text


def native_turn_text(reasoning: str, content: str, calls: list[tuple[str, dict]]) -> str:
    """Render a structured turn in the XML syntax the episode loop parses."""
    if not calls:
        return answer_text(content, reasoning)
    blocks = []
    for name, arguments in calls:
        parameters = "".join(
            f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
        )
        blocks.append(f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>")
    return f"<think>\n{reasoning}\n</think>\n\n" + "\n".join(blocks) + "<|im_end|>"


def _parse_arguments(raw) -> dict:
    """Arguments arrive as a JSON string; a malformed one stays visible.

    Returning the raw text under a single key lets schema validation turn it
    into a typed ``invalid_tool_call`` observation, which is a model fault the
    episode should record, not an adapter crash.
    """
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"malformed_arguments": str(raw)}
    return parsed if isinstance(parsed, dict) else {"malformed_arguments": str(raw)}


def _reasoning_from(message: dict) -> tuple[str | None, str]:
    """(reasoning or None when hidden, content) from a response message."""
    content = message.get("content") or ""
    for key in ("reasoning_content", "reasoning"):
        if key in message and message[key] is not None:
            return str(message[key]), content
    if "<think>" in content:
        # Some servers leave the think block inline in the content.
        reasoning, remainder = split_reasoning(content)
        return reasoning, remainder
    return None, content


def _reasoning_tokens_from(usage: dict | None) -> int | None:
    if not usage:
        return None
    details = usage.get("completion_tokens_details") or {}
    for candidate in (details.get("reasoning_tokens"), usage.get("reasoning_tokens")):
        if candidate is not None:
            return int(candidate)
    return None


def turn_from_response(response: dict, config: TeacherConfig) -> TurnResult:
    """Convert one chat-completions response into the loop's TurnResult."""
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TeacherError(f"malformed teacher response: {exc}") from exc

    reasoning, content = _reasoning_from(message)
    if reasoning is None:
        if config.require_reasoning:
            raise ReasoningHidden(
                f"{config.model} returned a turn without reasoning_content; set "
                "require_reasoning=False only if the endpoint is known to omit empty reasoning"
            )
        reasoning = ""

    calls = [
        (call["function"]["name"], _parse_arguments(call["function"].get("arguments")))
        for call in message.get("tool_calls") or []
    ]
    finish = choice.get("finish_reason")
    if finish == "content_filter":
        raise TeacherError(f"{config.model} stopped on content_filter")
    usage = response.get("usage") or {}
    return TurnResult(
        text=native_turn_text(reasoning, content, calls),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        fault="output_truncated" if finish == "length" else None,
        reasoning_tokens=_reasoning_tokens_from(usage),
    )


# ---------------------------------------------------------------------------
# Transport


def http_transport(config: TeacherConfig, sleep: Callable[[float], None] = time.sleep) -> Transport:
    """POST to ``{base_url}/chat/completions`` with bounded retries."""
    api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
    if config.api_key_env and not api_key:
        raise TeacherError(f"{config.api_key_env} is not set; the teacher endpoint needs it")
    url = config.base_url.rstrip("/") + "/chat/completions"

    def send(payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as reply:
                    return json.loads(reply.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = TeacherError(f"HTTP {exc.code} from {url}: {detail}")
                if exc.code not in RETRYABLE_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = TeacherError(f"{type(exc).__name__} talking to {url}: {exc}")
            if attempt < config.max_retries:
                sleep(2.0**attempt)
        raise last_error or TeacherError("teacher request failed")

    return send


# ---------------------------------------------------------------------------
# Policy


def request_payload(config: TeacherConfig, messages: list[dict], seed: int | None = None) -> dict:
    payload = {
        "model": config.model,
        "messages": to_api_messages(messages, config.send_reasoning_back),
        "tools": TOOLS,
        "tool_choice": config.tool_choice,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort
    payload.update(config.extra_body)
    return payload


def teacher_policy_factory(config: TeacherConfig, transport: Transport | None = None):
    """A ``(task, seed) -> Policy`` factory the collector and gate accept."""
    send = transport or http_transport(config)

    def factory(task, seed: int) -> Policy:
        del task

        def policy(messages: list[dict]) -> TurnResult:
            return turn_from_response(send(request_payload(config, messages, seed)), config)

        return policy

    return factory


PROBE_MESSAGES = [
    {"role": "developer", "content": "Inspect before editing and run tests before completing."},
    {"role": "user", "content": "Read src/app.py and tell me what it does. Start by reading the file."},
]


def probe_teacher(config: TeacherConfig, transport: Transport | None = None) -> dict:
    """One round trip that reports whether the endpoint can be a teacher.

    A teacher is usable when it answers a tool-bearing prompt with a native
    tool call and exposes its reasoning; the probe says which of those hold
    before any budget is spent on episodes.
    """
    send = transport or http_transport(config)
    probe_config = TeacherConfig(**{**config.__dict__, "require_reasoning": False})
    response = send(request_payload(probe_config, PROBE_MESSAGES, seed=0))
    message = response["choices"][0]["message"]
    reasoning, _ = _reasoning_from(message)
    turn = turn_from_response(response, probe_config)
    from .parsing import parse_tool_calls  # local: keeps the module's import graph flat

    _, calls = parse_tool_calls(turn.text)
    return {
        "model": response.get("model", config.model),
        "endpoint": config.base_url,
        "tool_call_returned": bool(calls),
        "first_tool": calls[0]["function"]["name"] if calls else None,
        "reasoning_visible": reasoning is not None,
        "reasoning_tokens_reported": turn.reasoning_tokens is not None,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usable": bool(calls) and (reasoning is not None or not config.require_reasoning),
    }
