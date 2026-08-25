"""Parameterised task fixtures for bootstrap corpus generation.

Each family is a tiny Python repository with a planted bug, a gold fix and
real unit tests. Variants change identifiers, constants and phrasing so rows
are not clones, while the bug class per family stays fixed. Every artifact a
trajectory will observe (file contents, patches, test output) is produced by
actually executing the fixture, never by writing observations by hand.

Deliberate pilot limits, recorded in the corpus quality report: Python only,
single-file fixes, pytest as the only build system. Scale beyond the smoke
corpus needs real repositories per docs/data-strategy.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

MODULE_DIR = "src"
TESTS_DIR = "tests"

DEVELOPER_POOL = [
    "Inspect before editing and run tests before completing.",
    "Work only in the provided repository. Verify changes with the unit test profile.",
    "Make the smallest correct change and verify it with the unit tests.",
    "Investigate the failing behaviour first, then make the minimal edit and run the tests.",
]

ANSWER_DEVELOPER_POOL = [
    "Answer from repository evidence only. Do not edit any file.",
    "Inspect the repository and answer the question; make no changes.",
]


@dataclass(frozen=True)
class FixtureTask:
    """One concrete task variant, with every state a trajectory shape needs."""

    family: str
    variant: int
    module_path: str
    tests_path: str
    buggy_module: str
    fixed_module: str
    strong_tests: str
    weak_tests: str | None
    test_addition: str | None
    partial_module: str | None
    request: str
    developer: str
    search_query: str
    answer_question: str
    answer_text: str
    read_reasoning: str
    bug_reasoning: str
    verify_reasoning: str
    summary: str
    partial_reasoning: str | None
    recovery_reasoning: str | None
    supports_test_author: bool = True


def _rng(family: str, variant: int) -> random.Random:
    seed = int(hashlib.sha256(f"{family}:{variant}".encode()).hexdigest(), 16) % 2**32
    return random.Random(seed)


def _tests(module_name: str, imports: str, functions: list[str]) -> str:
    return f"from {MODULE_DIR}.{module_name} import {imports}\n\n" + "\n".join(functions) + "\n"


def _family_bounds(variant: int) -> FixtureTask:
    rng = _rng("bounds", variant)
    fn = rng.choice(["clamp", "bound", "limit_to", "restrict"])
    module_name = rng.choice(["bounds", "limits", "numeric"])
    low, span = rng.randint(-5, 5), rng.randint(4, 12)
    high = low + span
    inside = low + span // 2
    below, above = low - rng.randint(1, 9), high + rng.randint(1, 9)
    header = f'def {fn}(value, low, high):\n    """Return value forced into the closed range [low, high]."""\n'
    buggy = header + "    return value\n"
    fixed = header + "    return max(low, min(high, value))\n"
    partial = header + "    return max(low, value)\n"
    strong = _tests(module_name, fn, [
        f"def test_inside_range_unchanged():\n    assert {fn}({inside}, {low}, {high}) == {inside}\n",
        f"def test_below_uses_lower_bound():\n    assert {fn}({below}, {low}, {high}) == {low}\n",
        f"def test_above_uses_upper_bound():\n    assert {fn}({above}, {low}, {high}) == {high}\n",
    ])
    weak = _tests(module_name, fn, [
        f"def test_inside_range_unchanged():\n    assert {fn}({inside}, {low}, {high}) == {inside}\n",
    ])
    addition = (
        f"\ndef test_out_of_range_values_use_bounds():\n"
        f"    assert {fn}({below}, {low}, {high}) == {low}\n"
        f"    assert {fn}({above}, {low}, {high}) == {high}\n"
    )
    return FixtureTask(
        family="bounds", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() returns out-of-range values unchanged. Fix it so both bounds are enforced, then run the unit tests.",
            f"Values outside [low, high] pass straight through {fn}(). Make it respect both bounds and verify with the tests.",
            f"The range helper {fn}() ignores its bounds. Fix it minimally and run the test suite.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query=f"def {fn}",
        answer_question=f"Which function forces a value into its configured range, and in which file is it defined? Inspect and answer without editing.",
        answer_text=f"The range is enforced by `{fn}()` in `{MODULE_DIR}/{module_name}.py`; it clamps with `max(low, min(high, value))`.",
        read_reasoning=f"I should read {MODULE_DIR}/{module_name}.py to see what {fn} currently does.",
        bug_reasoning=f"{fn} returns the value untouched, so neither bound is applied. Clamping with max/min is the minimal fix.",
        verify_reasoning="I should run the unit tests to confirm both bounds now hold.",
        summary=f"Fixed {fn}() to clamp with max(low, min(high, value)) and verified the unit tests pass.",
        partial_reasoning=f"{fn} ignores its bounds; applying the lower bound with max should fix it.",
        recovery_reasoning="The upper-bound test still fails: I only applied the lower bound. The value must also be capped with min against high.",
    )


def _family_csv_fields(variant: int) -> FixtureTask:
    rng = _rng("csv_fields", variant)
    fn = rng.choice(["parse_fields", "split_record", "parse_row"])
    module_name = rng.choice(["fields", "records", "rows"])
    sep = rng.choice([",", ";", "|"])
    a, b = rng.choice([("alpha", "beta"), ("x1", "y2"), ("north", "south"), ("id", "name")])
    header = (
        f'def {fn}(text, sep="{sep}"):\n'
        f'    """Split one record into fields; an empty record has no fields."""\n'
    )
    buggy = header + "    return text.split(sep)\n"
    fixed = header + '    if text == "":\n        return []\n    return text.split(sep)\n'
    partial = header + "    return [field for field in text.split(sep) if field]\n"
    strong = _tests(module_name, fn, [
        f'def test_basic_split():\n    assert {fn}("{a}{sep}{b}") == ["{a}", "{b}"]\n',
        f'def test_empty_record_has_no_fields():\n    assert {fn}("") == []\n',
        f'def test_inner_empty_fields_survive():\n    assert {fn}("{a}{sep}{sep}{b}") == ["{a}", "", "{b}"]\n',
    ])
    weak = _tests(module_name, fn, [
        f'def test_basic_split():\n    assert {fn}("{a}{sep}{b}") == ["{a}", "{b}"]\n',
    ])
    addition = f'\ndef test_empty_record_has_no_fields():\n    assert {fn}("") == []\n'
    return FixtureTask(
        family="csv_fields", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() returns [''] for an empty record instead of an empty list. Fix that edge case and run the unit tests.",
            f"An empty input record should produce no fields, but {fn}() yields one empty field. Fix it minimally and verify.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query=f"def {fn}",
        answer_question=f"Which function splits a raw record into fields, and what separator does it default to? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` splits records and defaults to the separator \"{sep}\".",
        read_reasoning=f"I should inspect {fn} to see how the empty record is handled.",
        bug_reasoning='"".split(sep) yields [""], so the empty record needs an explicit branch that returns [].',
        verify_reasoning="I should run the unit tests to cover the empty-record and inner-empty cases.",
        summary=f"Added the explicit empty-record branch to {fn}() and verified all unit tests pass.",
        partial_reasoning="Filtering out falsy fields should remove the spurious empty entry.",
        recovery_reasoning="Filtering broke legitimate inner empty fields; only the fully empty record should return [], so I need an explicit branch instead.",
    )


def _family_retry(variant: int) -> FixtureTask:
    rng = _rng("retry", variant)
    fn = rng.choice(["call_with_retry", "retry_call", "with_retries"])
    module_name = rng.choice(["retry", "attempts", "resilience"])
    attempts = rng.randint(3, 5)
    header = (
        f'def {fn}(operation, attempts):\n'
        f'    """Call operation up to attempts times; re-raise the last ValueError."""\n'
        f"    last_error = None\n"
    )
    buggy = header + (
        "    for _ in range(attempts - 1):\n"
        "        try:\n            return operation()\n"
        "        except ValueError as error:\n            last_error = error\n"
        "    raise last_error\n"
    )
    fixed = header + (
        "    for _ in range(attempts):\n"
        "        try:\n            return operation()\n"
        "        except ValueError as error:\n            last_error = error\n"
        "    raise last_error\n"
    )
    partial = header + (
        "    for _ in range(attempts):\n"
        "        try:\n            return operation()\n"
        "        except ValueError:\n            pass\n"
        "    raise last_error\n"
    )
    strong = _tests(module_name, fn, [
        (
            "def test_success_on_final_attempt():\n"
            "    calls = []\n"
            "    def flaky():\n"
            f"        calls.append(1)\n        if len(calls) < {attempts}:\n"
            '            raise ValueError("not yet")\n        return "ok"\n'
            f'    assert {fn}(flaky, {attempts}) == "ok"\n'
            f"    assert len(calls) == {attempts}\n"
        ),
        (
            "def test_exhausted_raises_value_error():\n"
            "    import pytest\n"
            "    def always_fails():\n"
            '        raise ValueError("permanent")\n'
            "    with pytest.raises(ValueError):\n"
            f"        {fn}(always_fails, {attempts})\n"
        ),
    ])
    weak = _tests(module_name, fn, [
        (
            "def test_first_attempt_success():\n"
            f'    assert {fn}(lambda: "ok", {attempts}) == "ok"\n'
        ),
    ])
    addition = (
        "\ndef test_success_on_final_attempt():\n"
        "    calls = []\n"
        "    def flaky():\n"
        f"        calls.append(1)\n        if len(calls) < {attempts}:\n"
        '            raise ValueError("not yet")\n        return "ok"\n'
        f'    assert {fn}(flaky, {attempts}) == "ok"\n'
    )
    return FixtureTask(
        family="retry", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() gives up one attempt early, so an operation that succeeds on its final try still fails. Fix the loop and run the tests.",
            f"With attempts={attempts}, {fn}() only ever calls the operation {attempts - 1} times. Fix the off-by-one and verify with the unit tests.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="range\\(attempts",
        answer_question="Which function implements the retry loop, and which exception type does it retry on? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` implements the retry loop and retries on ValueError.",
        read_reasoning=f"I should read the retry loop in {fn} and count its iterations.",
        bug_reasoning=f"range(attempts - 1) drops the final attempt, so the loop must run range(attempts) to use every try.",
        verify_reasoning="I should run the unit tests to confirm the final attempt is now used.",
        summary=f"Fixed the off-by-one in {fn}() so all attempts are used, and verified the unit tests pass.",
        partial_reasoning="Widening the loop to range(attempts) should cover the final try; the except body can be simplified while I am here.",
        recovery_reasoning="The exhaustion test now dies raising None because I stopped recording the last error. I need to keep the last_error assignment and only widen the range.",
    )


def _family_lru_cache(variant: int) -> FixtureTask:
    rng = _rng("lru_cache", variant)
    cls = rng.choice(["SmallCache", "TinyCache", "BoundedCache"])
    module_name = rng.choice(["cache", "storage", "memo"])
    capacity = rng.randint(2, 3)
    keys = rng.choice([("a", "b", "c", "d"), ("k1", "k2", "k3", "k4"), ("red", "green", "blue", "gold")])
    def cache_body(eviction: str) -> str:
        return (
            f'class {cls}:\n'
            f'    """Insertion-ordered cache evicting the oldest key at capacity."""\n\n'
            f"    def __init__(self, capacity):\n"
            f"        self.capacity = capacity\n"
            "        self._data = {}\n"
            f"        self._order = []\n\n"
            f"    def put(self, key, value):\n"
            f"        if key in self._data:\n"
            f"            self._order.remove(key)\n"
            f"        elif len(self._data) >= self.capacity:\n"
            f"{eviction}"
            f"        self._data[key] = value\n"
            f"        self._order.append(key)\n\n"
            f"    def get(self, key):\n"
            f"        return self._data.get(key)\n"
        )

    buggy = cache_body("            evicted = self._order.pop()\n            del self._data[evicted]\n")
    fixed = cache_body("            evicted = self._order.pop(0)\n            del self._data[evicted]\n")
    partial = cache_body("            evicted = self._order.pop(0)\n")
    k0, k1, k2, _ = keys
    strong = _tests(module_name, cls, [
        (
            "def test_evicts_oldest_key():\n"
            f"    cache = {cls}({capacity})\n"
            + "".join(f'    cache.put("{key}", {index})\n' for index, key in enumerate(keys[: capacity + 1]))
            + f'    assert cache.get("{k0}") is None\n'
            + f'    assert cache.get("{keys[capacity]}") == {capacity}\n'
        ),
        (
            "def test_capacity_is_respected():\n"
            f"    cache = {cls}({capacity})\n"
            + "".join(f'    cache.put("{key}", {index})\n' for index, key in enumerate(keys[: capacity + 1]))
            + f"    present = [key for key in {list(keys[: capacity + 1])!r} if cache.get(key) is not None]\n"
            + f"    assert len(present) == {capacity}\n"
        ),
        (
            "def test_updating_existing_key_does_not_evict():\n"
            f"    cache = {cls}({capacity})\n"
            + "".join(f'    cache.put("{key}", 0)\n' for key in keys[:capacity])
            + f'    cache.put("{k0}", 9)\n'
            + f'    assert cache.get("{k0}") == 9\n'
            + f'    assert cache.get("{k1}") == 0\n'
        ),
    ])
    weak = _tests(module_name, cls, [
        (
            "def test_put_then_get():\n"
            f"    cache = {cls}({capacity})\n"
            f'    cache.put("{k0}", 1)\n'
            f'    assert cache.get("{k0}") == 1\n'
        ),
    ])
    addition = (
        "\ndef test_evicts_oldest_key():\n"
        f"    cache = {cls}({capacity})\n"
        + "".join(f'    cache.put("{key}", {index})\n' for index, key in enumerate(keys[: capacity + 1]))
        + f'    assert cache.get("{k0}") is None\n'
    )
    return FixtureTask(
        family="lru_cache", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{cls} evicts the most recently inserted key instead of the oldest one. Fix the eviction and run the unit tests.",
            f"At capacity, {cls}.put drops the newest entry rather than the oldest. Make eviction insertion-ordered and verify with tests.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="_order\\.pop",
        answer_question=f"Which class implements the bounded cache and which method performs eviction? Inspect and answer without editing.",
        answer_text=f"`{cls}` in `{MODULE_DIR}/{module_name}.py` is the bounded cache; eviction happens inside its `put()` method.",
        read_reasoning=f"I should read {cls}.put to see which end of the order list is evicted.",
        bug_reasoning="pop() takes the newest key from the order list; the oldest entry is index 0, so eviction must use pop(0).",
        verify_reasoning="I should run the unit tests to confirm oldest-first eviction.",
        summary=f"Changed {cls}.put to evict the oldest key with pop(0) and verified the unit tests pass.",
        partial_reasoning="Eviction should take index 0 from the order list instead of the end.",
        recovery_reasoning="The capacity test still fails: I evict from the order list but never delete the entry from the data dict, so the evicted key remains readable. Both structures must drop it.",
    )


def _family_slugify(variant: int) -> FixtureTask:
    rng = _rng("slugify", variant)
    fn = rng.choice(["slugify", "to_slug", "make_slug"])
    module_name = rng.choice(["slug", "urls", "naming"])
    word_a, word_b = rng.choice([("Hello", "World"), ("Release", "Notes"), ("Data", "Pipeline")])
    header = f'def {fn}(text):\n    """Lower-case text into single-dash-separated url slugs."""\n'
    buggy = header + '    return text.strip().lower().replace(" ", "-")\n'
    fixed = header + (
        '    slug = text.strip().lower().replace("_", "-").replace(" ", "-")\n'
        '    while "--" in slug:\n'
        '        slug = slug.replace("--", "-")\n'
        '    return slug.strip("-")\n'
    )
    partial = header + (
        '    slug = text.strip().lower().replace(" ", "-")\n'
        '    while "--" in slug:\n'
        '        slug = slug.replace("--", "-")\n'
        '    return slug.strip("-")\n'
    )
    lower_a, lower_b = word_a.lower(), word_b.lower()
    strong = _tests(module_name, fn, [
        f'def test_simple_words():\n    assert {fn}("{word_a} {word_b}") == "{lower_a}-{lower_b}"\n',
        f'def test_collapses_repeated_separators():\n    assert {fn}("{word_a}  {word_b}") == "{lower_a}-{lower_b}"\n',
        f'def test_underscores_become_dashes():\n    assert {fn}("{word_a}_{word_b}") == "{lower_a}-{lower_b}"\n',
        f'def test_no_leading_or_trailing_dash():\n    assert {fn}(" {word_a} ") == "{lower_a}"\n',
    ])
    weak = _tests(module_name, fn, [
        f'def test_simple_words():\n    assert {fn}("{word_a} {word_b}") == "{lower_a}-{lower_b}"\n',
    ])
    addition = (
        f'\ndef test_messy_input_is_normalised():\n'
        f'    assert {fn}("{word_a}  {word_b}") == "{lower_a}-{lower_b}"\n'
        f'    assert {fn}("{word_a}_{word_b}") == "{lower_a}-{lower_b}"\n'
    )
    return FixtureTask(
        family="slugify", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() leaves doubled dashes and raw underscores in slugs. Normalise them to single dashes and run the unit tests.",
            f"Slugs from {fn}() keep repeated separators and underscores. Fix the normalisation minimally and verify with tests.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query=f"def {fn}",
        answer_question="Which function builds url slugs, and what characters does it currently replace? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` builds slugs; it lower-cases and converts spaces and underscores to single dashes.",
        read_reasoning=f"I should read {fn} to see which separators it already handles.",
        bug_reasoning="Only single spaces are replaced, so runs of separators and underscores leak through. Collapsing repeated dashes and mapping underscores fixes it.",
        verify_reasoning="I should run the unit tests across the messy-input cases.",
        summary=f"Normalised {fn}() to collapse repeated dashes, map underscores and trim edges; all unit tests pass.",
        partial_reasoning="Collapsing doubled dashes after the space replacement should normalise the slugs.",
        recovery_reasoning="The underscore test still fails: underscores never became dashes in the first place, so they must be replaced before collapsing.",
    )


def _family_intervals(variant: int) -> FixtureTask:
    rng = _rng("intervals", variant)
    fn = rng.choice(["overlaps", "intervals_overlap", "ranges_intersect"])
    module_name = rng.choice(["intervals", "ranges", "spans"])
    base = rng.randint(0, 6)
    width = rng.randint(2, 5)
    a0, a1 = base, base + width
    b0, b1 = a1, a1 + width
    header = (
        f'def {fn}(a_start, a_end, b_start, b_end):\n'
        f'    """True when the half-open intervals [start, end) share any point."""\n'
    )
    buggy = header + "    return a_start <= b_end and b_start <= a_end\n"
    fixed = header + "    return a_start < b_end and b_start < a_end\n"
    partial = header + "    return a_start < b_end and b_start <= a_end\n"
    strong = _tests(module_name, fn, [
        f"def test_disjoint_is_false():\n    assert {fn}({a0}, {a1}, {b1 + 1}, {b1 + 2}) is False\n",
        f"def test_touching_is_false():\n    assert {fn}({a0}, {a1}, {b0}, {b1}) is False\n",
        f"def test_touching_reversed_is_false():\n    assert {fn}({b0}, {b1}, {a0}, {a1}) is False\n",
        f"def test_partial_overlap_is_true():\n    assert {fn}({a0}, {a1 + 1}, {a1}, {b1}) is True\n",
        f"def test_nested_is_true():\n    assert {fn}({a0}, {b1}, {a0 + 1}, {a1}) is True\n",
    ])
    weak = _tests(module_name, fn, [
        f"def test_partial_overlap_is_true():\n    assert {fn}({a0}, {a1 + 1}, {a1}, {b1}) is True\n",
    ])
    addition = (
        f"\ndef test_touching_half_open_is_false():\n"
        f"    assert {fn}({a0}, {a1}, {b0}, {b1}) is False\n"
        f"    assert {fn}({b0}, {b1}, {a0}, {a1}) is False\n"
    )
    return FixtureTask(
        family="intervals", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() treats touching half-open intervals as overlapping. Fix the boundary comparisons and run the unit tests.",
            f"The docstring of {fn}() promises half-open semantics, but adjacent intervals report an overlap. Fix it minimally and verify.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="a_start",
        answer_question="Which function decides interval overlap, and are its intervals closed or half-open? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` decides overlap and uses half-open [start, end) intervals, so touching intervals do not overlap.",
        read_reasoning=f"I should read {fn} and compare its comparisons against the half-open contract in the docstring.",
        bug_reasoning="Half-open intervals share a point only under strict inequality on both sides; the <= comparisons wrongly count touching endpoints.",
        verify_reasoning="I should run the unit tests, especially both touching orientations.",
        summary=f"Made both comparisons in {fn}() strict to honour half-open semantics; the unit tests pass.",
        partial_reasoning="Switching the first comparison to strict should stop touching intervals from overlapping.",
        recovery_reasoning="One touching orientation still reports an overlap because the second comparison stayed <=; both sides of the conjunction must be strict.",
    )


def _family_config(variant: int) -> FixtureTask:
    rng = _rng("config", variant)
    fn = rng.choice(["read_int_setting", "int_from_env", "get_int_option"])
    module_name = rng.choice(["config", "settings", "options"])
    key = rng.choice(["TIMEOUT_S", "MAX_WORKERS", "BATCH_SIZE", "RETRY_LIMIT"])
    default = rng.randint(2, 30)
    present = default + rng.randint(1, 20)
    header = (
        f'def {fn}(env, key, default):\n'
        f'    """Read an integer setting; unset or empty values use the default."""\n'
    )
    buggy = header + "    raw = env.get(key)\n    return int(raw)\n"
    fixed = header + (
        "    raw = env.get(key)\n"
        '    if raw is None or raw == "":\n'
        "        return default\n"
        "    return int(raw)\n"
    )
    partial = header + (
        "    raw = env.get(key)\n"
        "    if raw is None:\n"
        "        return default\n"
        "    return int(raw)\n"
    )
    strong = _tests(module_name, fn, [
        f'def test_present_value_is_parsed():\n    assert {fn}({{"{key}": "{present}"}}, "{key}", {default}) == {present}\n',
        f'def test_missing_key_uses_default():\n    assert {fn}({{}}, "{key}", {default}) == {default}\n',
        f'def test_empty_value_uses_default():\n    assert {fn}({{"{key}": ""}}, "{key}", {default}) == {default}\n',
    ])
    weak = _tests(module_name, fn, [
        f'def test_present_value_is_parsed():\n    assert {fn}({{"{key}": "{present}"}}, "{key}", {default}) == {present}\n',
    ])
    addition = (
        f'\ndef test_unset_and_empty_use_default():\n'
        f'    assert {fn}({{}}, "{key}", {default}) == {default}\n'
        f'    assert {fn}({{"{key}": ""}}, "{key}", {default}) == {default}\n'
    )
    return FixtureTask(
        family="config", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() crashes when {key} is unset or empty instead of using the default. Fix it and run the unit tests.",
            f"Reading {key} through {fn}() raises for missing and empty values. Apply the documented default behaviour and verify.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="env\\.get",
        answer_question="Which function parses integer settings, and what does its contract say about unset values? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` parses integer settings; its docstring commits unset or empty values to the caller's default.",
        read_reasoning=f"I should read {fn} to see how missing and empty values reach int().",
        bug_reasoning="int(None) and int(\"\") both raise, so the unset and empty cases must return the default before parsing.",
        verify_reasoning="I should run the unit tests over the present, missing and empty cases.",
        summary=f"Guarded {fn}() so unset and empty values fall back to the default; the unit tests pass.",
        partial_reasoning="Returning the default when the key is missing should stop the crash.",
        recovery_reasoning="The empty-string case still fails: env.get returns \"\" rather than None there, so the guard must treat empty exactly like unset.",
    )


def _family_rpn_calc(variant: int) -> FixtureTask:
    rng = _rng("rpn_calc", variant)
    fn = rng.choice(["evaluate", "eval_rpn", "rpn"])
    module_name = rng.choice(["calc", "rpn", "arith"])
    m, n = rng.randint(5, 9), rng.randint(2, 4)
    def calc_body(minus: str, div: str) -> str:
        return (
            f'def {fn}(tokens):\n'
            f'    """Evaluate reverse Polish notation over floats."""\n'
            f"    stack = []\n"
            f"    for token in tokens:\n"
            '        if token in {"+", "-", "*", "/"}:\n'
            f"            a = stack.pop()\n"
            f"            b = stack.pop()\n"
            f'            if token == "+":\n                stack.append(b + a)\n'
            f'            elif token == "-":\n                stack.append({minus})\n'
            f'            elif token == "*":\n                stack.append(b * a)\n'
            f"            else:\n                stack.append({div})\n"
            f"        else:\n"
            f"            stack.append(float(token))\n"
            f"    return stack[-1]\n"
        )

    buggy = calc_body("a - b", "a / b")
    fixed = calc_body("b - a", "b / a")
    partial = calc_body("b - a", "a / b")
    strong = _tests(module_name, fn, [
        f'def test_addition():\n    assert {fn}(["{m}", "{n}", "+"]) == {float(m + n)}\n',
        f'def test_subtraction_operand_order():\n    assert {fn}(["{m}", "{n}", "-"]) == {float(m - n)}\n',
        f'def test_division_operand_order():\n    assert {fn}(["{m * n}", "{n}", "/"]) == {float(m)}\n',
        f'def test_mixed_expression():\n    assert {fn}(["{m}", "{n}", "-", "{n}", "*"]) == {float((m - n) * n)}\n',
    ])
    weak = _tests(module_name, fn, [
        f'def test_addition():\n    assert {fn}(["{m}", "{n}", "+"]) == {float(m + n)}\n',
        f'def test_multiplication():\n    assert {fn}(["{m}", "{n}", "*"]) == {float(m * n)}\n',
    ])
    addition = (
        f'\ndef test_non_commutative_operand_order():\n'
        f'    assert {fn}(["{m}", "{n}", "-"]) == {float(m - n)}\n'
        f'    assert {fn}(["{m * n}", "{n}", "/"]) == {float(m)}\n'
    )
    return FixtureTask(
        family="rpn_calc", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() computes subtraction and division with swapped operands. Fix the operand order and run the unit tests.",
            f'["{m}", "{n}", "-"] should evaluate to {float(m - n)} but {fn}() returns {float(n - m)}. Fix the stack order for the non-commutative operators and verify.',
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="stack\\.pop",
        answer_question="Which function evaluates RPN expressions, and in what order does it pop operands? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` evaluates RPN; it pops the second operand first, so the earlier stack entry must be the left-hand side.",
        read_reasoning=f"I should read the operator branch of {fn} and track which pop is the left operand.",
        bug_reasoning="The first pop is the right-hand operand, so subtraction and division must compute b - a and b / a.",
        verify_reasoning="I should run the unit tests over the non-commutative operators.",
        summary=f"Fixed operand order for subtraction and division in {fn}() and verified the unit tests pass.",
        partial_reasoning="Subtraction should combine the operands as b - a.",
        recovery_reasoning="Division still fails: it has the same swapped order as subtraction had, so it must also use b / a.",
    )


def _family_ring_buffer(variant: int) -> FixtureTask:
    rng = _rng("ring_buffer", variant)
    cls = rng.choice(["RingBuffer", "RollingWindow", "LastN"])
    module_name = rng.choice(["ring", "window", "recent"])
    capacity = rng.randint(2, 4)
    values = list(range(1, capacity + 3))
    body = (
        f'class {cls}:\n'
        f'    """Keep only the most recent capacity items, oldest first."""\n\n'
        f"    def __init__(self, capacity):\n"
        f"        self.capacity = capacity\n"
        f"        self._items = []\n\n"
        f"    def push(self, item):\n"
        f"        self._items.append(item)\n"
        f"{{trim}}"
        f"\n    def items(self):\n"
        f"        return list(self._items)\n"
    )
    buggy = body.format(trim="")
    fixed = body.format(trim="        if len(self._items) > self.capacity:\n            self._items.pop(0)\n")
    partial = body.format(trim="        if len(self._items) > self.capacity:\n            self._items.pop()\n")
    kept = values[-capacity:]
    strong = _tests(module_name, cls, [
        (
            "def test_keeps_only_most_recent():\n"
            f"    window = {cls}({capacity})\n"
            + "".join(f"    window.push({value})\n" for value in values)
            + f"    assert window.items() == {kept}\n"
        ),
        (
            "def test_under_capacity_keeps_everything():\n"
            f"    window = {cls}({capacity})\n"
            + "".join(f"    window.push({value})\n" for value in values[: capacity - 1])
            + f"    assert window.items() == {values[: capacity - 1]}\n"
        ),
    ])
    weak = _tests(module_name, cls, [
        (
            "def test_under_capacity_keeps_everything():\n"
            f"    window = {cls}({capacity})\n"
            + "".join(f"    window.push({value})\n" for value in values[: capacity - 1])
            + f"    assert window.items() == {values[: capacity - 1]}\n"
        ),
    ])
    addition = (
        "\ndef test_keeps_only_most_recent():\n"
        f"    window = {cls}({capacity})\n"
        + "".join(f"    window.push({value})\n" for value in values)
        + f"    assert window.items() == {kept}\n"
    )
    return FixtureTask(
        family="ring_buffer", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{cls} is documented to keep only the most recent items but grows without bound. Enforce the capacity and run the unit tests.",
            f"push() on {cls} never trims, so the window exceeds its capacity. Fix it to drop the oldest item and verify with the tests.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="def push",
        answer_question="Which class implements the rolling window, and from which end does it discard items? Inspect and answer without editing.",
        answer_text=f"`{cls}` in `{MODULE_DIR}/{module_name}.py` is the rolling window; at capacity it discards from the front, keeping the newest items.",
        read_reasoning=f"I should read {cls}.push to see whether the capacity is enforced.",
        bug_reasoning="push appends without trimming; when the list exceeds capacity the oldest entry at index 0 must be dropped.",
        verify_reasoning="I should run the unit tests for the over- and under-capacity behaviour.",
        summary=f"Enforced the capacity in {cls}.push by dropping the oldest item; the unit tests pass.",
        partial_reasoning="Trimming the list when it exceeds capacity should bound the window.",
        recovery_reasoning="The recency test fails: pop() removed the item I just pushed. The oldest entry lives at index 0, so the trim must use pop(0).",
    )


def _family_stats(variant: int) -> FixtureTask:
    rng = _rng("stats", variant)
    fn = rng.choice(["median", "middle_value", "central"])
    module_name = rng.choice(["stats", "summary", "metrics"])
    odd = sorted(rng.sample(range(1, 30), 5))
    even_base = sorted(rng.sample(range(1, 30), 4))
    while (even_base[1] + even_base[2]) % 2 == 0:
        even_base = sorted(rng.sample(range(1, 30), 4))
    even_expected = (even_base[1] + even_base[2]) / 2
    shuffled_odd = list(odd)
    rng.shuffle(shuffled_odd)
    header = f'def {fn}(values):\n    """Return the statistical median of a non-empty sequence."""\n'
    buggy = header + "    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n"
    fixed = header + (
        "    ordered = sorted(values)\n"
        "    middle = len(ordered) // 2\n"
        "    if len(ordered) % 2 == 1:\n"
        "        return ordered[middle]\n"
        "    return (ordered[middle - 1] + ordered[middle]) / 2\n"
    )
    partial = header + (
        "    ordered = sorted(values)\n"
        "    middle = len(ordered) // 2\n"
        "    if len(ordered) % 2 == 1:\n"
        "        return ordered[middle]\n"
        "    return (ordered[middle - 1] + ordered[middle]) // 2\n"
    )
    strong = _tests(module_name, fn, [
        f"def test_odd_count():\n    assert {fn}({shuffled_odd}) == {odd[2]}\n",
        f"def test_even_count_averages_middles():\n    assert {fn}({even_base}) == {even_expected}\n",
        f"def test_single_value():\n    assert {fn}([{odd[0]}]) == {odd[0]}\n",
    ])
    weak = _tests(module_name, fn, [
        f"def test_odd_count():\n    assert {fn}({shuffled_odd}) == {odd[2]}\n",
    ])
    addition = f"\ndef test_even_count_averages_middles():\n    assert {fn}({even_base}) == {even_expected}\n"
    return FixtureTask(
        family="stats", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f"{fn}() returns the upper middle element for even-sized input instead of the average of the two middles. Fix it and run the unit tests.",
            f"For {even_base} the median should be {even_expected}, but {fn}() returns {even_base[2]}. Fix the even-count branch and verify.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="sorted\\(values\\)",
        answer_question="Which function computes the median, and does it sort its input first? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` computes the median and sorts a copy of the input before selecting the middle.",
        read_reasoning=f"I should read {fn} and check how even-sized inputs are handled.",
        bug_reasoning="Indexing len//2 alone is only correct for odd counts; even counts need the mean of the two middle elements.",
        verify_reasoning="I should run the unit tests over odd, even and single-element inputs.",
        summary=f"Added the even-count branch to {fn}() so the two middles are averaged; the unit tests pass.",
        partial_reasoning="Averaging the two middle elements should fix the even-count case.",
        recovery_reasoning="The even-count test still fails: I used integer division for the average, which truncates .5 medians. It must be true division.",
    )


def _family_version_cmp(variant: int) -> FixtureTask:
    rng = _rng("version_cmp", variant)
    fn = rng.choice(["is_newer", "version_gt", "newer_than"])
    module_name = rng.choice(["versions", "semver", "release"])
    major = rng.randint(1, 4)
    # minor_small's digit must exceed minor_big's first digit so lexicographic
    # comparison is wrong for every variant, never coincidentally right.
    minor_small, minor_big = rng.randint(4, 9), rng.randint(10, 30)
    header = f'def {fn}(a, b):\n    """True when dotted numeric version a is newer than b."""\n'
    buggy = header + "    return a > b\n"
    fixed = header + (
        "    def parts(version):\n"
        '        return tuple(int(part) for part in version.split("."))\n'
        "    return parts(a) > parts(b)\n"
    )
    partial = header + (
        '    return int(a.split(".")[0]) > int(b.split(".")[0])\n'
    )
    v_big = f"{major}.{minor_big}.0"
    v_small = f"{major}.{minor_small}.0"
    strong = _tests(module_name, fn, [
        f'def test_numeric_component_comparison():\n    assert {fn}("{v_big}", "{v_small}") is True\n',
        f'def test_not_newer_reversed():\n    assert {fn}("{v_small}", "{v_big}") is False\n',
        f'def test_equal_is_not_newer():\n    assert {fn}("{v_big}", "{v_big}") is False\n',
        f'def test_major_bump():\n    assert {fn}("{major + 1}.0.0", "{v_big}") is True\n',
    ])
    weak = _tests(module_name, fn, [
        f'def test_major_bump():\n    assert {fn}("{major + 1}.0.0", "{major}.0.0") is True\n',
    ])
    addition = (
        f'\ndef test_multi_digit_components_compare_numerically():\n'
        f'    assert {fn}("{v_big}", "{v_small}") is True\n'
        f'    assert {fn}("{v_small}", "{v_big}") is False\n'
    )
    return FixtureTask(
        family="version_cmp", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=weak, test_addition=addition, partial_module=partial,
        request=rng.choice([
            f'{fn}() compares versions as strings, so "{v_small}" ranks above "{v_big}". Compare components numerically and run the unit tests.',
            f"Version ordering in {fn}() is lexicographic and breaks on multi-digit components. Fix it minimally and verify with the tests.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query=f"def {fn}",
        answer_question="Which function orders release versions, and what format does it expect? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` orders releases and expects dotted numeric versions such as \"{v_big}\".",
        read_reasoning=f"I should read {fn} to confirm how the version strings are compared.",
        bug_reasoning='String comparison ranks "9" above "10"; the components must be split on dots and compared as integer tuples.',
        verify_reasoning="I should run the unit tests, especially the multi-digit component case.",
        summary=f"Switched {fn}() to integer tuple comparison over the dotted components; the unit tests pass.",
        partial_reasoning="Comparing the major components as integers should restore numeric ordering.",
        recovery_reasoning="Versions sharing a major still compare wrongly because I ignored the remaining components; every dotted component must join the integer tuple.",
    )


def _family_dedupe(variant: int) -> FixtureTask:
    rng = _rng("dedupe", variant)
    fn = rng.choice(["unique", "dedupe", "distinct"])
    module_name = rng.choice(["dedupe", "collections_util", "sequence"])
    pool = rng.choice([["delta", "beta", "alpha", "gamma"], ["west", "north", "east", "south"], ["zeta", "eta", "theta", "iota"]])
    first_seen = [pool[0], pool[1], pool[2], pool[3]]
    with_dupes = [pool[0], pool[1], pool[0], pool[2], pool[1], pool[3]]
    header = f'def {fn}(items):\n    """Remove duplicates while keeping first-seen order."""\n'
    buggy = header + "    return list(set(items))\n"
    fixed = header + "    return list(dict.fromkeys(items))\n"
    partial = header + "    return sorted(set(items))\n"
    strong = _tests(module_name, fn, [
        f"def test_first_seen_order_is_kept():\n    assert {fn}({with_dupes}) == {first_seen}\n",
        f"def test_duplicates_are_removed():\n    assert sorted({fn}({with_dupes})) == {sorted(first_seen)}\n",
        f"def test_empty_input():\n    assert {fn}([]) == []\n",
    ])
    return FixtureTask(
        family="dedupe", variant=variant,
        module_path=f"{MODULE_DIR}/{module_name}.py", tests_path=f"{TESTS_DIR}/test_{module_name}.py",
        buggy_module=buggy, fixed_module=fixed, strong_tests=strong,
        weak_tests=None, test_addition=None, partial_module=partial,
        request=rng.choice([
            f"{fn}() loses the input order because it round-trips through a set. Preserve first-seen order and run the unit tests.",
            f"Deduplication in {fn}() must keep first-seen order per its docstring, but set() destroys it. Fix minimally and verify.",
        ]),
        developer=rng.choice(DEVELOPER_POOL),
        search_query="set\\(items\\)",
        answer_question="Which function removes duplicates, and what ordering does its contract promise? Inspect and answer without editing.",
        answer_text=f"`{fn}()` in `{MODULE_DIR}/{module_name}.py` removes duplicates and promises first-seen order of the surviving items.",
        read_reasoning=f"I should read {fn} and check whether the promised ordering survives.",
        bug_reasoning="list(set(...)) discards arrival order; dict.fromkeys keeps insertion order while removing duplicates.",
        verify_reasoning="I should run the unit tests including the ordering case.",
        summary=f"Replaced the set round-trip in {fn}() with dict.fromkeys to keep first-seen order; the unit tests pass.",
        partial_reasoning="Sorting the set at least makes the output deterministic and might satisfy the tests.",
        recovery_reasoning="Sorted order is not first-seen order, so the test still fails; dict.fromkeys preserves arrival order while deduplicating.",
        supports_test_author=False,
    )


FAMILY_BUILDERS = {
    "bounds": _family_bounds,
    "csv_fields": _family_csv_fields,
    "retry": _family_retry,
    "lru_cache": _family_lru_cache,
    "slugify": _family_slugify,
    "intervals": _family_intervals,
    "config": _family_config,
    "rpn_calc": _family_rpn_calc,
    "ring_buffer": _family_ring_buffer,
    "stats": _family_stats,
    "version_cmp": _family_version_cmp,
    "dedupe": _family_dedupe,
}

VARIANTS_PER_FAMILY = 17


def iter_tasks(variants_per_family: int = VARIANTS_PER_FAMILY):
    """Yield every fixture task in deterministic order."""
    for family in FAMILY_BUILDERS:
        for variant in range(variants_per_family):
            yield FAMILY_BUILDERS[family](variant)
