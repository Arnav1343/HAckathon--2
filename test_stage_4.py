"""
Test suite for Stage 4 — Compiler-Grade Entropy Freeze.
Tests validate_prompt_pack against all 14 validation checks.
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Stage_4 import validate_prompt_pack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ARCH = {
    "files": [
        {"path": "main.py", "responsibility": "Entry point", "depends_on": ["transaction_parser.py", "transaction_processor.py", "report_generator.py"]},
        {"path": "transaction_parser.py", "responsibility": "Parse lines", "depends_on": []},
        {"path": "transaction_processor.py", "responsibility": "Process", "depends_on": ["transaction_parser.py"]},
        {"path": "report_generator.py", "responsibility": "Report", "depends_on": ["transaction_processor.py"]},
    ]
}

SAMPLE_SPEC = {"objective": "Transaction tool", "inputs": ["file"], "outputs": ["report"],
               "functional_requirements": ["parse", "process", "report"], "constraints": []}

ARCH_PATHS = {f["path"] for f in SAMPLE_ARCH["files"]}


def make_valid_prompt_for(path: str) -> str:
    """Generate a minimal valid prompt for a given file path."""
    if path == "transaction_parser.py":
        return (
            'FILE PURPOSE: Parse each line by splitting on comma delimiter. '
            'Exactly 3 components: date, user, amount. Strip whitespace. Cast amount to float. '
            'REQUIRED FUNCTION SIGNATURES: parse_transaction_line(line: str) -> dict[str, str | float]. '
            'RETURN SCHEMA: Returns dict with exactly 3 keys. '
            'Keys: {"date": str, "user": str, "amount": float}. '
            'The dictionary must contain exactly 3 keys. No additional keys are allowed. '
            'INPUT/OUTPUT REQUIREMENTS: Accepts one line string. Returns dict with keys date, user, amount. '
            'DEPENDENCY RULES: This file has no dependencies. '
            'REQUIRED IMPORTS: No imports required. '
            'WHAT NOT TO DO: No extra functions. No architecture changes. No logging. '
            'No validation. No optimization. No new modules. No prohibited libraries. '
            'No defensive try/except blocks.'
        )
    elif path == "transaction_processor.py":
        return (
            'FILE PURPOSE: Iterate through list of transaction dicts. Sum amount values. '
            'Count transactions. Find user with highest cumulative amount. If tied, use first occurrence. '
            'REQUIRED FUNCTION SIGNATURES: process_transactions(transactions: list[dict[str, str | float]]) -> dict[str, int | float | str]. '
            'RETURN SCHEMA: Returns dict with exactly 3 keys. '
            'Keys: {"total_amount": float, "count": int, "top_user": str}. '
            'The dictionary must contain exactly 3 keys. No additional keys are allowed. '
            'All float values must be formatted using exactly 2 decimal places. Use format(value, ".2f"). '
            'INPUT/OUTPUT REQUIREMENTS: Accepts list of dicts with keys date, user, amount. '
            'Returns summary dict with keys total_amount, count, top_user. '
            'DEPENDENCY RULES: Depends on transaction_parser.py. '
            'REQUIRED IMPORTS: from transaction_parser import parse_transaction_line. '
            'WHAT NOT TO DO: No extra functions. No architecture changes. No logging. '
            'No validation. No optimization. No new modules. No prohibited libraries. '
            'No defensive try/except blocks.'
        )
    elif path == "report_generator.py":
        return (
            'FILE PURPOSE: Build plain text report from summary dict. '
            'Line 1: "Transaction Report". Line 2: 26 dashes. '
            'Line 3: "Total: " followed by total_amount with 2 decimal places using format(value, ".2f"). '
            'Line 4: "Count: " followed by count. Line 5: "Top User: " followed by top_user. '
            'No trailing newline. '
            'REQUIRED FUNCTION SIGNATURES: generate_report(summary: dict[str, int | float | str]) -> str. '
            'RETURN SCHEMA: Returns formatted str. Exact line order: header, separator, total, count, top_user. '
            'Separator is exactly 26 dashes. No trailing newline. '
            'All float values must be formatted using exactly 2 decimal places. Use format(value, ".2f"). '
            'INPUT/OUTPUT REQUIREMENTS: Accepts dict with keys total_amount, count, top_user. Returns str. '
            'DEPENDENCY RULES: Depends on transaction_processor.py. '
            'REQUIRED IMPORTS: from transaction_processor import process_transactions. '
            'WHAT NOT TO DO: No extra functions. No architecture changes. No logging. '
            'No validation. No optimization. No new modules. No prohibited libraries. '
            'No defensive try/except blocks.'
        )
    elif path == "main.py":
        return (
            'FILE PURPOSE: Entry point. Read file path from command-line argument. '
            'Open file, read lines, parse each, process all, generate report, print to stdout. '
            'REQUIRED FUNCTION SIGNATURES: main() -> None. '
            'RETURN SCHEMA: No return value. '
            'INPUT/OUTPUT REQUIREMENTS: Command-line argument: file path. Output: printed report to stdout. '
            'DEPENDENCY RULES: Depends on transaction_parser.py, transaction_processor.py, report_generator.py. '
            'REQUIRED IMPORTS: from transaction_parser import parse_transaction_line. '
            'from transaction_processor import process_transactions. '
            'from report_generator import generate_report. '
            'EXECUTION FLOW: 1. Read file_path from command-line args. '
            '2. Open file_path, read lines into raw_lines. '
            '3. Call parse_transaction_line(line) for each line, collect into transactions. '
            '4. Call process_transactions(transactions), store in summary. '
            '5. Call generate_report(summary), store in report_text. '
            '6. Print report_text. '
            'WHAT NOT TO DO: No extra functions. No architecture changes. No logging. '
            'No validation. No optimization. No new modules. No prohibited libraries. '
            'No defensive try/except blocks.'
        )
    return ""


def make_valid_pack() -> dict:
    """Build a fully valid prompt_pack."""
    return {
        "prompt_pack": [
            {"path": p, "implementation_prompt": make_valid_prompt_for(p)}
            for p in ARCH_PATHS
        ]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_pack_passes():
    """Happy path: fully compliant pack passes validation."""
    print("Testing valid pack passes...")
    pack = make_valid_pack()
    result = validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
    assert result is not None
    print("  PASS")
    return True


def test_missing_section_rejected():
    """Reject prompt missing a mandatory section."""
    print("Testing missing section rejected...")
    pack = make_valid_pack()
    # Remove RETURN SCHEMA from one prompt
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace("RETURN SCHEMA:", "DATA SHAPE:")
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted missing section")
        return False
    except ValueError as e:
        assert "missing mandatory section" in str(e)
    print("  PASS")
    return True


def test_section_order_enforced():
    """Reject prompt with sections out of order."""
    print("Testing section order enforced...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            # Swap RETURN SCHEMA and REQUIRED FUNCTION SIGNATURES
            p = entry["implementation_prompt"]
            p = p.replace("REQUIRED FUNCTION SIGNATURES:", "ZZTEMP:")
            p = p.replace("RETURN SCHEMA:", "REQUIRED FUNCTION SIGNATURES:")
            p = p.replace("ZZTEMP:", "RETURN SCHEMA:")
            entry["implementation_prompt"] = p
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted out-of-order sections")
        return False
    except ValueError as e:
        assert "out of order" in str(e)
    print("  PASS")
    return True


def test_bare_dict_return_rejected():
    """Reject bare -> dict without type parameters."""
    print("Testing bare dict return rejected...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "-> dict[str, str | float]", "-> dict"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted bare -> dict")
        return False
    except ValueError as e:
        assert "bare" in str(e).lower()
    print("  PASS")
    return True


def test_no_additional_keys_required():
    """Reject if RETURN SCHEMA lacks 'no additional keys' language."""
    print("Testing no-additional-keys enforcement...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "No additional keys are allowed.", ""
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted without 'no additional keys'")
        return False
    except ValueError as e:
        assert "no additional keys" in str(e).lower()
    print("  PASS")
    return True


def test_exactly_n_keys_required():
    """Reject if RETURN SCHEMA lacks 'exactly' keyword."""
    print("Testing 'exactly N keys' enforcement...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "exactly 3 keys", "3 keys"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted without 'exactly'")
        return False
    except ValueError as e:
        assert "exactly" in str(e).lower()
    print("  PASS")
    return True


def test_prohibition_parity():
    """Reject if any baseline prohibition is missing."""
    print("Testing prohibition parity...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "No defensive try/except blocks.", ""
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted missing prohibition")
        return False
    except ValueError as e:
        assert "baseline prohibition" in str(e).lower()
    print("  PASS")
    return True


def test_dependency_crosscheck():
    """Reject if main.py says 'does not depend' despite having deps."""
    print("Testing dependency cross-check...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "main.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "Depends on transaction_parser.py, transaction_processor.py, report_generator.py",
                "This file does not depend on any other files"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted false dependency claim")
        return False
    except ValueError as e:
        assert "does not depend" in str(e).lower()
    print("  PASS")
    return True


def test_import_bare_rejected():
    """Reject bare 'import X' in REQUIRED IMPORTS."""
    print("Testing bare import rejected...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_processor.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "from transaction_parser import parse_transaction_line",
                "import transaction_parser"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted bare import")
        return False
    except ValueError as e:
        assert "bare" in str(e).lower()
    print("  PASS")
    return True


def test_import_alias_rejected():
    """Reject 'as' alias in REQUIRED IMPORTS."""
    print("Testing import alias rejected...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_processor.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "from transaction_parser import parse_transaction_line",
                "from transaction_parser import parse_transaction_line as ptl"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted alias import")
        return False
    except ValueError as e:
        assert "alias" in str(e).lower()
    print("  PASS")
    return True


def test_execution_flow_required_for_entry():
    """Reject entry-point file missing EXECUTION FLOW."""
    print("Testing execution flow required for entry-point...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "main.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "EXECUTION FLOW:", "STEPS:"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted entry-point without EXECUTION FLOW")
        return False
    except ValueError as e:
        assert "execution flow" in str(e).lower()
    print("  PASS")
    return True


def test_tie_breaking_required():
    """Reject ranking logic without tie-breaking rule."""
    print("Testing tie-breaking enforcement...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_processor.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "If tied, use first occurrence.", ""
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted ranking without tie-breaking")
        return False
    except ValueError as e:
        assert "tie" in str(e).lower()
    print("  PASS")
    return True


def test_vague_adjective_rejected():
    """Reject expanded vague adjectives."""
    print("Testing vague adjective rejection (expanded)...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "Parse each line", "Parse each line in a straightforward manner"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted vague adjective 'straightforward'")
        return False
    except ValueError as e:
        assert "vague adjective" in str(e).lower()
    print("  PASS")
    return True


def test_word_count_no_minimum():
    """Short but dense prompts should pass (no minimum)."""
    print("Testing no word count minimum...")
    # A very short but valid prompt should pass
    # We test this implicitly — our valid prompts are already minimal
    pack = make_valid_pack()
    result = validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
    assert result is not None
    print("  PASS")
    return True


def test_meta_language_rejected():
    """Reject meta-language terms."""
    print("Testing meta-language rejection...")
    pack = make_valid_pack()
    for entry in pack["prompt_pack"]:
        if entry["path"] == "transaction_parser.py":
            entry["implementation_prompt"] = entry["implementation_prompt"].replace(
                "Parse each line", "In this pipeline stage, parse each line"
            )
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted meta-language")
        return False
    except ValueError as e:
        assert "meta-language" in str(e).lower()
    print("  PASS")
    return True


def test_missing_file_rejected():
    """Reject if not all architecture paths have prompts."""
    print("Testing missing file rejected...")
    pack = make_valid_pack()
    pack["prompt_pack"] = [e for e in pack["prompt_pack"] if e["path"] != "report_generator.py"]
    try:
        validate_prompt_pack(pack, ARCH_PATHS, SAMPLE_SPEC, SAMPLE_ARCH)
        print("  FAIL: Accepted missing file")
        return False
    except ValueError as e:
        assert "missing implementation prompts" in str(e).lower()
    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Running Stage 4 Compiler-Grade Freeze Tests\n")

    tests = [
        test_valid_pack_passes,
        test_missing_section_rejected,
        test_section_order_enforced,
        test_bare_dict_return_rejected,
        test_no_additional_keys_required,
        test_exactly_n_keys_required,
        test_prohibition_parity,
        test_dependency_crosscheck,
        test_import_bare_rejected,
        test_import_alias_rejected,
        test_execution_flow_required_for_entry,
        test_tie_breaking_required,
        test_vague_adjective_rejected,
        test_word_count_no_minimum,
        test_meta_language_rejected,
        test_missing_file_rejected,
    ]

    passed = 0
    for test in tests:
        try:
            if test(): passed += 1
        except Exception as e:
            print(f"  CRASH: {test.__name__} -> {e}")

    print(f"\nResults: {passed}/{len(tests)} passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
