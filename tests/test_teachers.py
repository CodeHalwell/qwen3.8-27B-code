"""Contracts for the teacher adapter: a larger model driving the same harness.

No network anywhere here. The transport is a fake that returns canned
chat-completions responses, which is exactly the seam the adapter exposes
for this purpose; the HTTP layer is tested against a patched urlopen.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from qwen3_8_27b_code import fixtures, policies, teachers
from qwen3_8_27b_code.collection import collect
from qwen3_8_27b_code.episodes import run_episode
from qwen3_8_27b_code.parsing import parse_tool_calls
from qwen3_8_27b_code.tasks import gold_patch, materialise, task_from_fixture

CONFIG = teachers.TeacherConfig(model="teacher/test", base_url="http://teacher.invalid/v1")


def response(*, reasoning="Thinking.", content="", calls=(), finish="stop", usage=None, hide_reasoning=False):
    message = {"role": "assistant", "content": content}
    if not hide_reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments},
            }
            for index, (name, arguments) in enumerate(calls)
        ]
    return {
        "model": "teacher/test",
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 40},
    }


def scripted_transport(responses, seen=None):
    queue = list(responses)

    def send(payload):
        if seen is not None:
            seen.append(payload)
        if not queue:
            raise RuntimeError("scripted transport exhausted")
        return queue.pop(0)

    return send


def teacher_like(task, seed):
    """A transport that behaves like a competent teacher on this task."""
    del seed
    return scripted_transport(
        [
            response(reasoning="I should read the module first.", calls=[("read_file", {"path": task.module_path})]),
            response(reasoning="The fix is a one-line change.", calls=[("apply_patch", {"patch": gold_patch(task)})]),
            response(reasoning="Verify.", calls=[("run_tests", {"profile": "unit"})]),
            response(reasoning="All green.", content="Fixed and verified."),
        ]
    )


# --------------------------------------------------------------------------
# Outgoing messages


def test_outgoing_messages_thread_tool_call_ids_and_fold_the_developer_role():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    with materialise(task) as workspace:
        episode = run_episode(workspace.harness, policies.gold(task, 0), request=task.request, developer=task.developer)

    api = teachers.to_api_messages(episode.messages)
    assert api[0]["role"] == "system" and api[0]["content"] == task.developer
    assert api[1] == {"role": "user", "content": task.request}
    assistant = [message for message in api if message["role"] == "assistant"]
    tools = [message for message in api if message["role"] == "tool"]
    call_ids = [call["id"] for message in assistant for call in message.get("tool_calls", [])]
    assert call_ids == ["call_0001", "call_0002", "call_0003"]
    assert [message["tool_call_id"] for message in tools] == call_ids
    assert [message["name"] for message in tools] == ["read_file", "apply_patch", "run_tests"]
    # Arguments travel as JSON strings, as the protocol requires.
    first_call = assistant[0]["tool_calls"][0]["function"]
    assert json.loads(first_call["arguments"]) == {"path": task.module_path}
    assert assistant[0]["reasoning_content"]
    # The final answer carries no tool calls and its content.
    assert "tool_calls" not in assistant[-1] and assistant[-1]["content"]

    without = teachers.to_api_messages(episode.messages, send_reasoning_back=False)
    assert not any("reasoning_content" in message for message in without)


def test_tool_observation_without_a_call_is_an_adapter_error():
    with pytest.raises(teachers.TeacherError):
        teachers.to_api_messages([{"role": "user", "content": "x"}, {"role": "tool", "name": "read_file", "content": "y"}])


# --------------------------------------------------------------------------
# Incoming responses


def test_structured_response_renders_to_native_text_the_loop_parses():
    turn = teachers.turn_from_response(
        response(reasoning="Look first.", calls=[("read_file", {"path": "src/a.py"})]), CONFIG
    )
    reasoning, calls = parse_tool_calls(turn.text)
    assert reasoning == "Look first."
    assert calls == [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "src/a.py"}}}]
    assert turn.fault is None
    assert (turn.prompt_tokens, turn.completion_tokens) == (100, 40)
    assert turn.reasoning_tokens is None

    several = teachers.turn_from_response(
        response(calls=[("read_file", {"path": "a"}), ("search", {"query": "def x"})]), CONFIG
    )
    assert [call["function"]["name"] for call in parse_tool_calls(several.text)[1]] == ["read_file", "search"]

    answer = teachers.turn_from_response(response(reasoning="Done.", content="Fixed it."), CONFIG)
    assert parse_tool_calls(answer.text) == ("Done.", [])
    assert answer.text.startswith("<think>\nDone.\n</think>\n\nFixed it.")


def test_reasoning_tokens_come_from_usage_details_when_present():
    usage = {"prompt_tokens": 1, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 33}}
    assert teachers.turn_from_response(response(usage=usage), CONFIG).reasoning_tokens == 33
    flat = {"prompt_tokens": 1, "completion_tokens": 50, "reasoning_tokens": 12}
    assert teachers.turn_from_response(response(usage=flat), CONFIG).reasoning_tokens == 12


def test_hidden_reasoning_is_refused_unless_explicitly_allowed():
    hidden = response(hide_reasoning=True, calls=[("read_file", {"path": "src/a.py"})])
    with pytest.raises(teachers.ReasoningHidden):
        teachers.turn_from_response(hidden, CONFIG)
    lenient = teachers.TeacherConfig(**{**CONFIG.__dict__, "require_reasoning": False})
    assert parse_tool_calls(teachers.turn_from_response(hidden, lenient).text)[0] == ""


def test_inline_think_block_counts_as_visible_reasoning():
    inline = response(hide_reasoning=True, content="<think>\nplan\n</think>\n\nThe answer.")
    turn = teachers.turn_from_response(inline, CONFIG)
    assert parse_tool_calls(turn.text) == ("plan", [])


def test_truncated_and_filtered_turns_are_reported_not_hidden():
    truncated = teachers.turn_from_response(response(finish="length"), CONFIG)
    assert truncated.fault == "output_truncated"
    with pytest.raises(teachers.TeacherError):
        teachers.turn_from_response(response(finish="content_filter"), CONFIG)
    with pytest.raises(teachers.TeacherError):
        teachers.turn_from_response({"choices": []}, CONFIG)


def test_malformed_arguments_become_a_typed_invalid_call_in_the_loop():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    transport = scripted_transport(
        [
            response(calls=[("read_file", "{not json")]),
            response(content="Giving up."),
        ]
    )
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            teachers.teacher_policy_factory(CONFIG, transport)(task, 0),
            request=task.request,
            developer=task.developer,
        )
    observations = [message["content"] for message in episode.messages if message["role"] == "tool"]
    assert observations[0].startswith("invalid_tool_call: read_file:")
    assert episode.invalid_tool_calls == 1


# --------------------------------------------------------------------------
# Through the collector


def test_teacher_attempts_are_collected_verified_and_labelled():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    seen: list[dict] = []

    # Build the factory by hand so every episode gets a fresh scripted transport.
    def factory(task, seed):
        transport = teacher_like(task, seed)

        def recording(payload):
            seen.append(payload)
            return transport(payload)

        return teachers.teacher_policy_factory(CONFIG, recording)(task, seed)

    result = collect(
        [task], factory, attempts_per_task=1, policy_label=CONFIG.label, source=CONFIG.label,
        provenance={"teacher": {"model": CONFIG.model}},
    )
    assert result.report()["rejections"] == {}
    row = result.rows[0]
    assert row["source"] == "teacher:teacher/test"
    assert row["provenance"]["policy"] == "teacher:teacher/test"
    assert row["provenance"]["teacher"] == {"model": "teacher/test"}
    assert row["verification"]["all_required_tests_pass"] is True
    assistant = [message for message in row["messages"] if message["role"] == "assistant"]
    assert all(message["reasoning_content"] for message in assistant)
    # Every request carried the deployment tool schema and the seed.
    assert all(payload["tools"] == teachers.TOOLS and payload["seed"] == 3407 for payload in seen)
    assert seen[-1]["messages"][-1]["role"] == "tool"


def test_hidden_reasoning_is_an_infrastructure_failure_not_a_model_failure():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))

    def factory(task, seed):
        transport = scripted_transport([response(hide_reasoning=True, calls=[("read_file", {"path": task.module_path})])])
        return teachers.teacher_policy_factory(CONFIG, transport)(task, seed)

    result = collect([task], factory, attempts_per_task=1)
    report = result.report()
    assert report["infrastructure_failures"] == 1
    assert report["rejections"] == {"infrastructure_failure": 1}
    assert "reasoning_content" in result.attempts[0].episode.final_text


# --------------------------------------------------------------------------
# Configuration, payloads, probe and transport


def test_presets_and_request_payload():
    config = teachers.TeacherConfig.from_preset("moonshot", "kimi-test", reasoning_effort="low", extra_body={"thinking": {"type": "enabled"}})
    assert config.base_url == "https://api.moonshot.ai/v1"
    assert config.api_key_env == "MOONSHOT_API_KEY"
    payload = teachers.request_payload(config, [{"role": "user", "content": "hi"}], seed=7)
    assert payload["model"] == "kimi-test" and payload["seed"] == 7
    assert payload["reasoning_effort"] == "low" and payload["thinking"] == {"type": "enabled"}
    assert payload["tools"] == teachers.TOOLS
    assert "reasoning_effort" not in teachers.request_payload(CONFIG, [{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError, match="unknown preset"):
        teachers.TeacherConfig.from_preset("nope", "x")
    for preset in teachers.PRESETS.values():
        assert preset.base_url.startswith(("http://", "https://"))


def test_probe_reports_whether_the_endpoint_can_teach():
    usable = teachers.probe_teacher(CONFIG, scripted_transport([response(calls=[("read_file", {"path": "src/app.py"})])]))
    assert usable["usable"] is True
    assert usable["first_tool"] == "read_file" and usable["reasoning_visible"] is True
    mute = teachers.probe_teacher(CONFIG, scripted_transport([response(hide_reasoning=True, calls=[("read_file", {"path": "src/app.py"})])]))
    assert mute["usable"] is False and mute["reasoning_visible"] is False
    prose = teachers.probe_teacher(CONFIG, scripted_transport([response(content="It is an app.")]))
    assert prose["usable"] is False and prose["tool_call_returned"] is False


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_transport_retries_transient_errors_and_raises_on_the_rest(monkeypatch):
    monkeypatch.setenv("TEACHER_KEY", "secret")
    config = teachers.TeacherConfig(model="m", base_url="http://teacher.invalid/v1", api_key_env="TEACHER_KEY", max_retries=2)
    calls = []
    outcomes = [
        urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"later")),
        _Reply(json.dumps(response()).encode()),
    ]

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teachers.urllib.request, "urlopen", fake_urlopen)
    send = teachers.http_transport(config, sleep=lambda seconds: None)
    assert send({"model": "m"})["choices"]
    assert calls[0][0] == "http://teacher.invalid/v1/chat/completions"
    assert calls[0][1] == "Bearer secret" and len(calls) == 2

    outcomes.append(urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"bad key")))
    with pytest.raises(teachers.TeacherError, match="HTTP 401"):
        send({"model": "m"})
    assert len(calls) == 3  # not retried

    monkeypatch.delenv("TEACHER_KEY")
    with pytest.raises(teachers.TeacherError, match="TEACHER_KEY"):
        teachers.http_transport(config)
