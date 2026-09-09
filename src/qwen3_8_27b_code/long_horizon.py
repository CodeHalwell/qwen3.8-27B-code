"""Multi-file, longer-horizon tasks for training and the held-out gate.

The single-file families in ``fixtures.py`` and ``tasks.py`` exercise the
inspect/edit/test loop in three to six tool calls, which is the short band of
docs/evaluation.md. Nothing there needs the agent to hold two files in mind at
once, decide which of several failing tests points where, or make more than
one edit before verification. These families do.

Every family plants one defect in each of two modules. The second module
imports the first, the visible tests cover both, and an integration path runs
through both, so a fix to either file alone leaves the suite red: the agent
has to find both causes, edit twice (or once across files) and verify. A gold
run takes seven tool calls, the medium band; a real policy that searches,
re-runs tests and recovers from a partial fix lands further along it.

Training families are graded by their visible tests only, like the fixtures,
and are what the collector samples from. Evaluation families carry hidden
verifiers like the single-file held-out families and are merged into the
gate's suite by ``tasks.evaluation_family_builders``. The two sets share no
family name, module name or bug class.
"""

from __future__ import annotations

import hashlib
import random

from .tasks import (
    DEVELOPER_POOL,
    MODULE_DIR,
    TESTS_DIR,
    AgentTask,
    HiddenCheck,
    hidden_package_script,
)


def _rng(kind: str, family: str, variant: int) -> random.Random:
    seed = int(hashlib.sha256(f"{kind}:{family}:{variant}".encode()).hexdigest(), 16) % 2**32
    return random.Random(seed)


def _task(
    kind: str,
    family: str,
    variant: int,
    rng: random.Random,
    request: str,
    buggy: dict[str, str],
    fixed: dict[str, str],
    tests: dict[str, str],
    imports: tuple[str, ...] = (),
    contract: str | None = None,
    regression: str | None = None,
) -> AgentTask:
    if set(buggy) != set(fixed):
        raise ValueError(f"{family}: buggy and fixed file sets differ")
    hidden: tuple[HiddenCheck, ...] = ()
    if kind == "eval":
        if contract is None or regression is None:
            raise ValueError(f"{family}: evaluation tasks need contract and regression checks")
        hidden = (
            HiddenCheck("contract", hidden_package_script(list(imports), contract)),
            HiddenCheck("no_regression", hidden_package_script(list(imports), regression)),
        )
    first_module = sorted(buggy)[0]
    first_test = sorted(tests)[0]
    return AgentTask(
        task_id=f"{'eval' if kind == 'eval' else 'train'}/{family}-{variant:03d}",
        family=family,
        variant=variant,
        request=request,
        developer=rng.choice(DEVELOPER_POOL),
        files={**buggy, **tests},
        hidden_checks=hidden,
        protected_paths=tuple(sorted(tests)),
        module_path=first_module,
        tests_path=first_test,
        reference_module=fixed[first_module],
        reference_files=dict(fixed),
    )


# ---------------------------------------------------------------------------
# Training families: sampled by the collector, graded by visible tests.


def _family_quantity_pipeline(variant: int) -> AgentTask:
    rng = _rng("train", "quantity_pipeline", variant)
    parse_mod = rng.choice(["quantity", "measure", "reading"])
    convert_mod = rng.choice(["units", "conversion", "scale"])
    parse_fn = rng.choice(["parse_quantity", "split_quantity", "read_quantity"])
    convert_fn = rng.choice(["to_base", "in_base_units", "as_base"])
    pipeline_fn = rng.choice(["normalise", "canonical_amount", "base_amount"])
    big, base, small = rng.choice([("kg", "g", "mg"), ("km", "m", "mm"), ("kl", "l", "ml")])
    upper = big.upper()

    parse_header = (
        f"def {parse_fn}(text):\n"
        f'    """Split "12 {upper} " into (12.0, "{big}"): the unit is trimmed and lower-cased."""\n'
    )
    parse_buggy = parse_header + '    number, _, unit = text.partition(" ")\n    return float(number), unit\n'
    parse_fixed = parse_header + (
        '    number, _, unit = text.strip().partition(" ")\n'
        "    return float(number), unit.strip().lower()\n"
    )
    pipeline_body = (
        f"\n\ndef {pipeline_fn}(text):\n"
        f'    """Parse a quantity string and express it in the base unit."""\n'
        f"    return {convert_fn}(*{parse_fn}(text))\n"
    )
    convert_header = (
        f"from {MODULE_DIR}.{parse_mod} import {parse_fn}\n\n"
        f'FACTORS = {{"{big}": 1000.0, "{base}": 1.0, "{small}": 0.001}}\n\n\n'
        f"def {convert_fn}(value, unit):\n"
        f'    """Convert value in unit to the base unit; an unknown unit raises ValueError."""\n'
    )
    convert_buggy = convert_header + "    return value * FACTORS.get(unit, 1.0)\n" + pipeline_body
    convert_fixed = convert_header + (
        "    if unit not in FACTORS:\n"
        '        raise ValueError(f"unknown unit: {unit}")\n'
        "    return value * FACTORS[unit]\n"
    ) + pipeline_body

    tests = {
        f"{TESTS_DIR}/test_{parse_mod}.py": (
            f"from {MODULE_DIR}.{parse_mod} import {parse_fn}\n\n\n"
            "def test_padded_upper_case_unit_is_normalised():\n"
            f'    assert {parse_fn}("2 {upper} ") == (2.0, "{big}")\n\n\n'
            "def test_plain_quantity():\n"
            f'    assert {parse_fn}("5 {base}") == (5.0, "{base}")\n'
        ),
        f"{TESTS_DIR}/test_{convert_mod}.py": (
            "import pytest\n\n"
            f"from {MODULE_DIR}.{convert_mod} import {convert_fn}, {pipeline_fn}\n\n\n"
            "def test_known_unit_scales():\n"
            f'    assert {convert_fn}(2, "{big}") == 2000.0\n\n\n'
            "def test_unknown_unit_is_rejected():\n"
            "    with pytest.raises(ValueError):\n"
            f'        {convert_fn}(1, "stone")\n\n\n'
            "def test_pipeline_handles_upper_case_units():\n"
            f'    assert {pipeline_fn}("3 {upper}") == 3000.0\n'
        ),
    }
    request = rng.choice([
        f'{pipeline_fn}("3 {upper}") returns 3.0 instead of 3000.0, and unknown units are silently accepted. '
        "Find both causes, fix them minimally and run the unit tests.",
        f"Quantities with upper-case or padded units go wrong through {parse_fn}(), and {convert_fn}() does not "
        "reject unknown units. Fix both modules and verify with the unit tests.",
    ])
    return _task(
        "train", "quantity_pipeline", variant, rng, request,
        buggy={f"{MODULE_DIR}/{parse_mod}.py": parse_buggy, f"{MODULE_DIR}/{convert_mod}.py": convert_buggy},
        fixed={f"{MODULE_DIR}/{parse_mod}.py": parse_fixed, f"{MODULE_DIR}/{convert_mod}.py": convert_fixed},
        tests=tests,
    )


def _family_inventory(variant: int) -> AgentTask:
    rng = _rng("train", "inventory", variant)
    stock_mod = rng.choice(["stock", "holdings", "stockroom"])
    report_mod = rng.choice(["reorder", "shortages", "topup"])
    cls = rng.choice(["Stock", "Holdings", "StockRoom"])
    build_fn = rng.choice(["from_counts", "stock_from", "build_stock"])
    low_fn = rng.choice(["low_stock", "below_threshold", "short_items"])
    restock_fn = rng.choice(["restock_list", "reorder_plan", "top_up_orders"])
    threshold = rng.randint(3, 6)
    batch = threshold + rng.randint(4, 8)

    stock_header = (
        f"class {cls}:\n"
        f'    """Named item counts; removing more than is held raises ValueError."""\n\n'
        "    def __init__(self):\n"
        "        self._counts = {}\n\n"
        "    def add(self, name, quantity):\n"
        "        self._counts[name] = self._counts.get(name, 0) + quantity\n\n"
        "    def remove(self, name, quantity):\n"
    )
    stock_tail = (
        "\n    def count(self, name):\n"
        "        return self._counts.get(name, 0)\n\n"
        "    def names(self):\n"
        "        return sorted(self._counts)\n"
    )
    stock_buggy = stock_header + "        self._counts[name] = self._counts.get(name, 0) - quantity\n" + stock_tail
    stock_fixed = stock_header + (
        "        held = self._counts.get(name, 0)\n"
        "        if quantity > held:\n"
        '            raise ValueError(f"cannot remove {quantity} {name}: only {held} held")\n'
        "        self._counts[name] = held - quantity\n"
    ) + stock_tail

    report_header = (
        f"from {MODULE_DIR}.{stock_mod} import {cls}\n\n\n"
        f"def {build_fn}(counts):\n"
        f'    """Build a {cls} from a name -> quantity mapping."""\n'
        f"    stock = {cls}()\n"
        "    for name, quantity in counts.items():\n"
        "        stock.add(name, quantity)\n"
        "    return stock\n\n\n"
        f"def {low_fn}(stock, threshold):\n"
        f'    """Names whose count is strictly below threshold, alphabetically."""\n'
    )
    restock_body = (
        f"\n\ndef {restock_fn}(stock, threshold, batch):\n"
        f'    """(name, quantity to order) pairs that bring every low item up to batch."""\n'
        f"    return [(name, batch - stock.count(name)) for name in {low_fn}(stock, threshold)]\n"
    )
    report_buggy = report_header + (
        "    return [name for name in stock.names() if stock.count(name) <= threshold]\n"
    ) + restock_body
    report_fixed = report_header + (
        "    return [name for name in stock.names() if stock.count(name) < threshold]\n"
    ) + restock_body

    tests = {
        f"{TESTS_DIR}/test_{stock_mod}.py": (
            "import pytest\n\n"
            f"from {MODULE_DIR}.{stock_mod} import {cls}\n\n\n"
            "def test_add_then_count():\n"
            f"    stock = {cls}()\n"
            '    stock.add("bolt", 4)\n'
            '    stock.add("bolt", 2)\n'
            '    assert stock.count("bolt") == 6\n\n\n'
            "def test_removing_more_than_held_is_rejected():\n"
            f"    stock = {cls}()\n"
            '    stock.add("nut", 2)\n'
            "    with pytest.raises(ValueError):\n"
            '        stock.remove("nut", 3)\n'
        ),
        f"{TESTS_DIR}/test_{report_mod}.py": (
            f"from {MODULE_DIR}.{report_mod} import {build_fn}, {low_fn}, {restock_fn}\n\n\n"
            "def test_item_at_the_threshold_is_not_low():\n"
            f'    stock = {build_fn}({{"washer": {threshold}, "bolt": {threshold - 2}}})\n'
            f'    assert {low_fn}(stock, {threshold}) == ["bolt"]\n\n\n'
            "def test_restock_brings_low_items_up_to_batch():\n"
            f'    stock = {build_fn}({{"bolt": 1, "washer": {threshold}}})\n'
            f'    assert {restock_fn}(stock, {threshold}, {batch}) == [("bolt", {batch - 1})]\n'
        ),
    }
    request = rng.choice([
        f"{cls}.remove() lets a count go negative instead of refusing, and {low_fn}() reports items that sit "
        "exactly at the threshold. Fix both and run the unit tests.",
        "The reorder report over-orders and the stock ledger accepts impossible removals. Investigate both "
        "modules, make the minimal fixes and verify with the unit tests.",
    ])
    return _task(
        "train", "inventory", variant, rng, request,
        buggy={f"{MODULE_DIR}/{stock_mod}.py": stock_buggy, f"{MODULE_DIR}/{report_mod}.py": report_buggy},
        fixed={f"{MODULE_DIR}/{stock_mod}.py": stock_fixed, f"{MODULE_DIR}/{report_mod}.py": report_fixed},
        tests=tests,
    )


# ---------------------------------------------------------------------------
# Evaluation families: hidden verifiers, never collected from.


def _family_ledger(variant: int) -> AgentTask:
    rng = _rng("eval", "ledger", variant)
    money_mod = rng.choice(["money", "amounts", "currency"])
    ledger_mod = rng.choice(["ledger", "statement", "account"])
    parse_fn = rng.choice(["parse_pence", "to_pence", "pence_from"])
    fee_fn = rng.choice(["apply_fee", "charge_fees", "with_fees"])
    settle_fn = rng.choice(["settle", "balance_after_fees", "net_balance"])
    fee = rng.choice([10, 25, 50])

    money_header = (
        f"def {parse_fn}(text):\n"
        f'    """Parse an amount such as "£12.50" or "-£0.99" into signed integer pence."""\n'
    )
    money_buggy = money_header + (
        '    digits = text.strip().lstrip("-£")\n'
        '    pounds, _, pence = digits.partition(".")\n'
        "    return int(pounds) * 100 + int(pence or 0)\n"
    )
    money_fixed = money_header + (
        "    cleaned = text.strip()\n"
        '    sign = -1 if cleaned.startswith("-") else 1\n'
        '    digits = cleaned.lstrip("-£")\n'
        '    pounds, _, pence = digits.partition(".")\n'
        "    return sign * (int(pounds) * 100 + int(pence or 0))\n"
    )
    settle_body = (
        f"\n\ndef {settle_fn}(entries, fee):\n"
        f'    """Parse the statement entries and return the balance after debit fees."""\n'
        f"    return sum({fee_fn}([{parse_fn}(entry) for entry in entries], fee))\n"
    )
    ledger_header = (
        f"from {MODULE_DIR}.{money_mod} import {parse_fn}\n\n\n"
        f"def {fee_fn}(amounts, fee):\n"
        f'    """Charge fee pence on every debit (negative amount); credits are untouched."""\n'
    )
    ledger_buggy = ledger_header + "    return [amount - fee for amount in amounts]\n" + settle_body
    ledger_fixed = ledger_header + (
        "    return [amount - fee if amount < 0 else amount for amount in amounts]\n"
    ) + settle_body

    tests = {
        f"{TESTS_DIR}/test_{money_mod}.py": (
            f"from {MODULE_DIR}.{money_mod} import {parse_fn}\n\n\n"
            "def test_positive_amount():\n"
            f'    assert {parse_fn}("£12.50") == 1250\n\n\n'
            "def test_negative_amount_keeps_its_sign():\n"
            f'    assert {parse_fn}("-£3.20") == -320\n'
        ),
        f"{TESTS_DIR}/test_{ledger_mod}.py": (
            f"from {MODULE_DIR}.{ledger_mod} import {fee_fn}, {settle_fn}\n\n\n"
            "def test_credits_are_not_charged():\n"
            f"    assert {fee_fn}([500, -200], {fee}) == [500, {-(200 + fee)}]\n\n\n"
            "def test_settle_charges_debits_only():\n"
            f'    assert {settle_fn}(["£10.00", "-£2.50"], {fee}) == {1000 - 250 - fee}\n'
        ),
    }
    contract = f'''
    assert {parse_fn}("-£0.99") == -99
    assert {parse_fn}("£0.05") == 5
    assert {fee_fn}([], {fee}) == []
    assert {fee_fn}([-100, -100], 5) == [-105, -105]
    assert {fee_fn}([300], 0) == [300]
    assert {settle_fn}(["-£1.00", "£1.00"], 50) == -50
    '''
    regression = f'''
    assert {parse_fn}("£12.50") == 1250
    assert {parse_fn}("£7") == 700
    assert {settle_fn}(["£1.00", "£2.00"], 0) == 300
    '''
    request = rng.choice([
        f"{settle_fn}() returns the wrong balance whenever a statement mixes credits and debits. Find every "
        "cause, fix them minimally and run the unit tests.",
        f"Two things are wrong in the ledger: {parse_fn}() drops the sign of a debit, and {fee_fn}() charges "
        "the fee to credits too. Fix both and verify with the unit tests.",
    ])
    return _task(
        "eval", "ledger", variant, rng, request,
        buggy={f"{MODULE_DIR}/{money_mod}.py": money_buggy, f"{MODULE_DIR}/{ledger_mod}.py": ledger_buggy},
        fixed={f"{MODULE_DIR}/{money_mod}.py": money_fixed, f"{MODULE_DIR}/{ledger_mod}.py": ledger_fixed},
        tests=tests,
        imports=(
            f"from {MODULE_DIR}.{money_mod} import {parse_fn}",
            f"from {MODULE_DIR}.{ledger_mod} import {fee_fn}, {settle_fn}",
        ),
        contract=contract,
        regression=regression,
    )


def _family_word_stats(variant: int) -> AgentTask:
    rng = _rng("eval", "word_stats", variant)
    tok_mod = rng.choice(["tokenise", "wordsplit", "lexer"])
    count_mod = rng.choice(["frequency", "ranking", "wordcount"])
    words_fn = rng.choice(["words", "tokens", "split_words"])
    top_fn = rng.choice(["top_words", "most_frequent", "rank_words"])
    summarise_fn = rng.choice(["summarise", "headline_words", "key_terms"])

    tok_header = (
        "import string\n\n\n"
        f"def {words_fn}(text):\n"
        f'    """Lower-case words with surrounding punctuation removed; empty tokens are dropped."""\n'
    )
    tok_buggy = tok_header + "    return [token.lower() for token in text.split()]\n"
    tok_fixed = tok_header + (
        "    cleaned = [token.strip(string.punctuation).lower() for token in text.split()]\n"
        "    return [token for token in cleaned if token]\n"
    )
    summarise_body = (
        f"\n\ndef {summarise_fn}(text, n):\n"
        f'    """The n most frequent words of text."""\n'
        f"    return {top_fn}({words_fn}(text), n)\n"
    )
    count_header = (
        "from collections import Counter\n\n"
        f"from {MODULE_DIR}.{tok_mod} import {words_fn}\n\n\n"
        f"def {top_fn}(tokens, n):\n"
        f'    """The n most frequent tokens; ties are broken alphabetically."""\n'
    )
    count_buggy = count_header + (
        "    return [token for token, _ in Counter(tokens).most_common(n)]\n"
    ) + summarise_body
    count_fixed = count_header + (
        "    counts = Counter(tokens)\n"
        "    ranked = sorted(counts, key=lambda token: (-counts[token], token))\n"
        "    return ranked[:n]\n"
    ) + summarise_body

    tests = {
        f"{TESTS_DIR}/test_{tok_mod}.py": (
            f"from {MODULE_DIR}.{tok_mod} import {words_fn}\n\n\n"
            "def test_punctuation_is_stripped():\n"
            f'    assert {words_fn}("Stop. Go!") == ["stop", "go"]\n\n\n'
            "def test_whitespace_runs_are_one_gap():\n"
            f'    assert {words_fn}("Two   words") == ["two", "words"]\n'
        ),
        f"{TESTS_DIR}/test_{count_mod}.py": (
            f"from {MODULE_DIR}.{count_mod} import {summarise_fn}, {top_fn}\n\n\n"
            "def test_ties_break_alphabetically():\n"
            f'    assert {top_fn}(["pear", "apple", "pear", "apple"], 1) == ["apple"]\n\n\n'
            "def test_summary_counts_words_not_punctuated_tokens():\n"
            f'    assert {summarise_fn}("Go go, stop.", 2) == ["go", "stop"]\n'
        ),
    }
    contract = f'''
    assert {words_fn}("") == []
    assert {words_fn}("Hello, hello!") == ["hello", "hello"]
    assert {top_fn}(["zeta", "yak", "yak", "zeta"], 2) == ["yak", "zeta"]
    assert {top_fn}([], 3) == []
    assert {summarise_fn}("Tea? Tea! tea. cake", 1) == ["tea"]
    '''
    regression = f'''
    assert {words_fn}("plain words") == ["plain", "words"]
    assert {top_fn}(["a", "a", "b"], 1) == ["a"]
    assert {summarise_fn}("one two two", 1) == ["two"]
    '''
    request = rng.choice([
        f"{summarise_fn}() ranks punctuated tokens as separate words and orders equal counts arbitrarily. "
        "Track down both causes, fix them minimally and run the unit tests.",
        f"{words_fn}() keeps punctuation attached to words and {top_fn}() ignores the documented alphabetical "
        "tie-break. Fix both modules and verify with the unit tests.",
    ])
    return _task(
        "eval", "word_stats", variant, rng, request,
        buggy={f"{MODULE_DIR}/{tok_mod}.py": tok_buggy, f"{MODULE_DIR}/{count_mod}.py": count_buggy},
        fixed={f"{MODULE_DIR}/{tok_mod}.py": tok_fixed, f"{MODULE_DIR}/{count_mod}.py": count_fixed},
        tests=tests,
        imports=(
            f"from {MODULE_DIR}.{tok_mod} import {words_fn}",
            f"from {MODULE_DIR}.{count_mod} import {summarise_fn}, {top_fn}",
        ),
        contract=contract,
        regression=regression,
    )


TRAINING_FAMILY_BUILDERS = {
    "quantity_pipeline": _family_quantity_pipeline,
    "inventory": _family_inventory,
}

EVALUATION_FAMILY_BUILDERS = {
    "ledger": _family_ledger,
    "word_stats": _family_word_stats,
}

TRAINING_VARIANTS_PER_FAMILY = 2


def training_tasks(variants_per_family: int = TRAINING_VARIANTS_PER_FAMILY) -> list[AgentTask]:
    """Multi-file training tasks for the collector, deterministic variants."""
    return [
        TRAINING_FAMILY_BUILDERS[family](variant)
        for family in TRAINING_FAMILY_BUILDERS
        for variant in range(variants_per_family)
    ]
