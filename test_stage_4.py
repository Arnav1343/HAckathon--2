"""
Test suite for Stage 4 — Code Generation.
Verifies AST-based structural validation, signature matching, and dependency guards.
"""

import sys
import os
import logging
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Stage_4 import validate_file_implementation, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockContext:
    def __init__(self, state=None):
        self._state = state or {}
        self._logger = logging.getLogger("SpecForge_Stage4_Test")

    def get(self, key):
        return self._state.get(key)

    def set(self, key, value):
        self._state[key] = value

    @property
    def logger(self):
        return self._logger


# ---------------------------------------------------------------------------
# validate_file_implementation tests
# ---------------------------------------------------------------------------

def test_validate_valid_contract_file():
    """Happy path: File matches Stage 3 contract perfectly."""
    print("Testing valid contract file...")
    code = """
import math

def calculate_area(radius: float) -> float:
    return math.pi * radius * radius
"""
    contract = {
        "path": "geo.py",
        "exports": [
            {"name": "calculate_area", "inputs": ["radius: float"], "returns": "float"}
        ]
    }
    deps = ["math"] # Stdlib allowed
    local_mods = {"geo"}
    contract_names = {"calculate_area"}
    
    assert validate_file_implementation(code, "geo.py", contract, deps, local_mods, contract_names) is True
    print("  PASS")
    return True


def test_validate_signature_mismatch():
    """Fail if param names or types don't match exactly."""
    print("Testing signature mismatch...")
    contract = {
        "path": "geo.py",
        "exports": [
            {"name": "f", "inputs": ["x: int"], "returns": "str"}
        ]
    }
    local_mods = {"geo"}
    contract_names = {"f"}

    # Wrong param name
    code1 = "def f(y: int) -> str: return str(y)"
    try:
        validate_file_implementation(code1, "geo.py", contract, [], local_mods, contract_names)
        print("  FAIL: Accepted wrong param name")
        return False
    except ValueError as e:
        assert "Signature mismatch" in str(e)

    # Wrong param type
    code2 = "def f(x: float) -> str: return str(x)"
    try:
        validate_file_implementation(code2, "geo.py", contract, [], local_mods, contract_names)
        print("  FAIL: Accepted wrong param type")
        return False
    except ValueError as e:
        assert "Signature mismatch" in str(e)

    # Wrong return type
    code3 = "def f(x: int) -> int: return x"
    try:
        validate_file_implementation(code3, "geo.py", contract, [], local_mods, contract_names)
        print("  FAIL: Accepted wrong return type")
        return False
    except ValueError as e:
        assert "Signature mismatch" in str(e)

    print("  PASS")
    return True


def test_validate_strict_lean():
    """Fail if extra top-level functions exist in a contract file."""
    print("Testing strict lean policy...")
    code = """
def contract_func() -> None:
    pass

def helper_func() -> None:
    pass
"""
    contract = {
        "path": "a.py",
        "exports": [{"name": "contract_func", "inputs": [], "returns": "None"}]
    }
    try:
        validate_file_implementation(code, "a.py", contract, [], {"a"}, {"contract_func"})
        print("  FAIL: Accepted extra top-level function")
        return False
    except ValueError as e:
        assert "Extra top-level functions" in str(e)
    
    print("  PASS")
    return True


def test_validate_main_file_logic():
    """Non-contract files can have helpers but no contract conflicts."""
    print("Testing main-file logic...")
    all_contracts = {"some_api_func"}
    
    # Valid main
    code_ok = "def main(): pass\ndef helper(): pass"
    assert validate_file_implementation(code_ok, "main.py", {"exports": []}, [], {"main"}, all_contracts) is True
    
    # Conflict with global contract
    code_bad = "def some_api_func(): pass"
    try:
        validate_file_implementation(code_bad, "main.py", {"exports": []}, [], {"main"}, all_contracts)
        print("  FAIL: Accepted conflict with global contract")
        return False
    except ValueError as e:
        assert "conflict with project-wide contracts" in str(e)
        
    print("  PASS")
    return True


def test_validate_dependency_guard():
    """Fail if local module imported but not in deps."""
    print("Testing dependency guard...")
    code = "import db_utils\ndef f() -> None: pass"
    contract = {"exports": [{"name": "f", "inputs": [], "returns": "None"}]}
    local_mods = {"f", "db_utils"}
    
    # Not in deps
    try:
        validate_file_implementation(code, "f.py", contract, [], local_mods, {"f"})
        print("  FAIL: Accepted undeclared local dependency")
        return False
    except ValueError as e:
        assert "Undeclared local dependency" in str(e)
        
    # In deps
    assert validate_file_implementation(code, "f.py", contract, ["db_utils.py"], local_mods, {"f"}) is True
    
    print("  PASS")
    return True


def test_validate_forbidden_structures():
    """Fail on Classes or AsyncFunctions."""
    print("Testing forbidden structures...")
    local_mods = {"a"}
    contract = {"exports": [{"name": "f", "inputs": [], "returns": "None"}]}
    
    # Class
    code_class = "class MyClass: pass\ndef f(): pass"
    try:
        validate_file_implementation(code_class, "a.py", contract, [], local_mods, {"f"})
        print("  FAIL: Accepted ClassDef")
        return False
    except ValueError as e:
        assert "Class" in str(e)

    # Async
    code_async = "async def f(): pass"
    try:
        validate_file_implementation(code_async, "a.py", contract, [], local_mods, {"f"})
        print("  FAIL: Accepted AsyncFunctionDef")
        return False
    except ValueError as e:
        assert "Async function" in str(e)

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Running Stage 4 Code Generation Tests\n")

    tests = [
        test_validate_valid_contract_file,
        test_validate_signature_mismatch,
        test_validate_strict_lean,
        test_validate_main_file_logic,
        test_validate_dependency_guard,
        test_validate_forbidden_structures
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
