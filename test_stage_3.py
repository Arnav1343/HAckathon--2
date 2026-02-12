"""
Test suite for Stage 3 — Interface Contracts.
"""

import sys
import os
import logging
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Stage_3 import validate_contracts, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockContext:
    def __init__(self, state=None):
        self._state = state or {}
        self._logger = logging.getLogger("SpecForge_Stage3_Test")

    def get(self, key):
        return self._state.get(key)

    def set(self, key, value):
        self._state[key] = value

    @property
    def logger(self):
        return self._logger


# ---------------------------------------------------------------------------
# validate_contracts tests
# ---------------------------------------------------------------------------

def test_validate_contracts_valid():
    """Happy path — valid contracts matching depended set."""
    print("Testing validate_contracts valid input...")
    depended = {"auth.py", "db_utils.py"}
    contracts = {
        "files": [
            {
                "path": "auth.py",
                "exports": [
                    {"name": "login", "inputs": ["user: str", "pwd: str"], "returns": "bool"}
                ]
            },
            {
                "path": "db_utils.py",
                "exports": [
                    {"name": "query_sync", "inputs": ["sql: str"], "returns": "list[dict]"}
                ]
            }
        ]
    }
    result = validate_contracts(contracts, depended)
    assert result == contracts
    print("  PASS")
    return True


def test_validate_contracts_minimality_exact_match():
    """Fail if contract set != depended set."""
    print("Testing validate_contracts exact-match minimality...")
    depended = {"a.py", "b.py"}
    
    # Missing a file
    missing = {"files": [{"path": "a.py", "exports": [{"name": "f", "inputs": [], "returns": "v"}]}]}
    try:
        validate_contracts(missing, depended)
        print("  FAIL: Accepted missing depended file")
        return False
    except ValueError as e:
        assert "Missing" in str(e)
    
    # Extra file
    extra = {
        "files": [
            {"path": "a.py", "exports": [{"name": "f", "inputs": [], "returns": "v"}]},
            {"path": "b.py", "exports": [{"name": "f", "inputs": [], "returns": "v"}]},
            {"path": "extra.py", "exports": [{"name": "f", "inputs": [], "returns": "v"}]}
        ]
    }
    try:
        validate_contracts(extra, depended)
        print("  FAIL: Accepted non-depended file")
        return False
    except ValueError as e:
        assert "not depended upon" in str(e)
        
    print("  PASS")
    return True


def test_validate_contracts_schema_strictness():
    """Fail on extraneous keys or empty exports."""
    print("Testing validate_contracts schema strictness...")
    depended = {"a.py"}
    
    # Extra root key
    try:
        validate_contracts({"files": [], "illegal": 1}, depended)
        print("  FAIL: Accepted extra root key")
        return False
    except ValueError as e:
        assert "exactly one key" in str(e)

    # Empty exports
    empty_exp = {"files": [{"path": "a.py", "exports": []}]}
    try:
        validate_contracts(empty_exp, depended)
        print("  FAIL: Accepted empty exports")
        return False
    except ValueError as e:
        assert "zero exports" in str(e)
        
    print("  PASS")
    return True


def test_validate_contracts_signature_regex():
    """Fail on invalid function names, inputs, or return formats."""
    print("Testing validate_contracts signature regex...")
    depended = {"a.py"}
    
    cases = [
        ("Bad Name", {"name": "generate-haiku", "inputs": [], "returns": "str"}),
        ("Bad Input", {"name": "f", "inputs": ["badFormat"], "returns": "str"}),
        ("Bad Return", {"name": "f", "inputs": [], "returns": "str with space"})
    ]
    
    for label, exp in cases:
        bad_contracts = {"files": [{"path": "a.py", "exports": [exp]}]}
        try:
            validate_contracts(bad_contracts, depended)
            print(f"  FAIL: Accepted {label}")
            return False
        except ValueError:
            pass

    print("  PASS")
    return True


def test_validate_contracts_uniqueness():
    """Fail on duplicate export names in one file."""
    print("Testing validate_contracts uniqueness...")
    depended = {"a.py"}
    dup = {
        "files": [
            {
                "path": "a.py",
                "exports": [
                    {"name": "f", "inputs": [], "returns": "v"},
                    {"name": "f", "inputs": [], "returns": "v"}
                ]
            }
        ]
    }
    try:
        validate_contracts(dup, depended)
        print("  FAIL: Accepted duplicate function names")
        return False
    except ValueError as e:
        assert "Duplicate export name" in str(e)
    
    print("  PASS")
    return True


def test_run_guards():
    """Test Stage 3 entry point guards."""
    print("Testing run() guards...")
    
    # Missing arch
    assert run(MockContext({})) is False
    
    # Malformed arch
    assert run(MockContext({"architecture": "not a dict"})) is False
    
    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Running Stage 3 Interface Contracts Tests\n")

    tests = [
        test_validate_contracts_valid,
        test_validate_contracts_minimality_exact_match,
        test_validate_contracts_schema_strictness,
        test_validate_contracts_signature_regex,
        test_validate_contracts_uniqueness,
        test_run_guards
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
