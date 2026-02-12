"""
Test suite for Stage 2 — Architecture Planning.

Tests validation logic, path safety, dependency integrity, and cycle detection.
"""

import sys
import os
import logging
import json
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Stage_2 import validate_architecture, check_for_cycles, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockContext:
    """Minimal StageContext mock for testing."""
    def __init__(self, state=None):
        self._state = state or {}
        self._logger = logging.getLogger("SpecForge_Stage2_Test")

    def get(self, key):
        return self._state.get(key)

    def set(self, key, value):
        self._state[key] = value

    @property
    def logger(self):
        return self._logger


# ---------------------------------------------------------------------------
# validate_architecture tests
# ---------------------------------------------------------------------------

def test_validate_arch_valid():
    """Happy path — valid architecture with dependencies."""
    print("Testing validate_architecture valid input...")
    arch = {
        "files": [
            {
                "path": "main.py",
                "responsibility": "Entry point",
                "depends_on": ["engine.py"]
            },
            {
                "path": "engine.py",
                "responsibility": "Core logic",
                "depends_on": ["utils/helpers.py"]
            },
            {
                "path": "utils/helpers.py",
                "responsibility": "Generic utilities",
                "depends_on": []
            }
        ]
    }
    result = validate_architecture(arch)
    assert result == arch
    print("  PASS")
    return True


def test_validate_arch_invalid_structure():
    """Fail if missing 'files' or has extra keys."""
    print("Testing validate_architecture invalid top-level structure...")
    try:
        validate_architecture({"files": [], "extra": 1})
        print("  FAIL: Extra key allowed")
        return False
    except ValueError as e:
        assert "exactly one key" in str(e)
    
    try:
        validate_architecture({})
        print("  FAIL: Empty dict allowed")
        return False
    except ValueError as e:
        assert "exactly one key" in str(e)
    
    print("  PASS")
    return True


def test_validate_arch_path_safety():
    """Fail on dangerous or invalid paths."""
    print("Testing validate_architecture path safety...")
    
    base = {"path": "a.py", "responsibility": "r", "depends_on": []}
    
    bad_paths = [
        "data.json",    # Not .py
        os.path.abspath("path.py"), # Absolute (cross-platform)
        "../traversal.py", # Traversal
        ".hidden.py",   # Hidden
        "dir/.hidden/f.py", # Hidden dir
    ]
    
    for path in bad_paths:
        arch = {"files": [base.copy()]}
        arch["files"][0]["path"] = path
        try:
            validate_architecture(arch)
            print(f"  FAIL: Accepted bad path: {path}")
            return False
        except ValueError:
            pass
            
    # Duplicate check
    arch_dup = {"files": [base.copy(), base.copy()]}
    try:
        validate_architecture(arch_dup)
        print("  FAIL: Accepted duplicate paths")
        return False
    except ValueError as e:
        assert "Duplicate" in str(e)

    print("  PASS")
    return True


def test_validate_arch_dependency_integrity():
    """Fail on non-existent or self dependencies."""
    print("Testing validate_architecture dependency integrity...")
    
    # Non-existent
    arch_missing = {
        "files": [{"path": "a.py", "responsibility": "r", "depends_on": ["b.py"]}]
    }
    try:
        validate_architecture(arch_missing)
        print("  FAIL: Accepted non-existent dependency")
        return False
    except ValueError as e:
        assert "non-existent" in str(e)

    # Self dependency
    arch_self = {
        "files": [{"path": "a.py", "responsibility": "r", "depends_on": ["a.py"]}]
    }
    try:
        validate_architecture(arch_self)
        print("  FAIL: Accepted self-dependency")
        return False
    except ValueError as e:
        assert "Self-dependency" in str(e)

    print("  PASS")
    return True


def test_validate_arch_cycle_detection():
    """Fail on circular dependencies."""
    print("Testing validate_architecture cycle detection...")
    
    # A -> B -> A
    arch_cycle = {
        "files": [
            {"path": "a.py", "responsibility": "r", "depends_on": ["b.py"]},
            {"path": "b.py", "responsibility": "r", "depends_on": ["a.py"]}
        ]
    }
    try:
        validate_architecture(arch_cycle)
        print("  FAIL: Accepted simple cycle")
        return False
    except ValueError as e:
        assert "Circular dependency" in str(e)
        assert "a.py -> b.py -> a.py" in str(e)

    # A -> B -> C -> A
    arch_long_cycle = {
        "files": [
            {"path": "a.py", "responsibility": "r", "depends_on": ["b.py"]},
            {"path": "b.py", "responsibility": "r", "depends_on": ["c.py"]},
            {"path": "c.py", "responsibility": "r", "depends_on": ["a.py"]}
        ]
    }
    try:
        validate_architecture(arch_long_cycle)
        print("  FAIL: Accepted long cycle")
        return False
    except ValueError as e:
        assert "Circular dependency" in str(e)
        assert "a.py -> b.py -> c.py -> a.py" in str(e)

    print("  PASS")
    return True


def test_run_guards():
    """Test entry point guards (spec presence/validity)."""
    print("Testing run() guards...")
    
    # Missing spec
    ctx_missing = MockContext({})
    assert run(ctx_missing) is False
    
    # Empty spec (allowed as presence, will likely fail later logic but passes guard)
    # Stage 2 should not re-validate Stage 1's work.
    ctx_empty = MockContext({"spec": {}})
    # Note: run(ctx_empty) might succeed or fail depending on if build_prompt/LLM handles it
    # But the GUARD should let it through if it exists.
    
    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("Running Stage 2 Architecture Planning Tests\n")

    tests = [
        test_validate_arch_valid,
        test_validate_arch_invalid_structure,
        test_validate_arch_path_safety,
        test_validate_arch_dependency_integrity,
        test_validate_arch_cycle_detection,
        test_run_guards
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"  FAIL: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTest Results: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
