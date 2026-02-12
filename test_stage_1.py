#!/usr/bin/env python3
"""
Test suite for Stage 1 — Specification Engine.

Tests validation logic, prompt construction, and run() guards.
LLM calls are not tested here (requires API key + network).
"""

import sys
import os
import logging
from unittest.mock import patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from Stage_1 import validate_spec, build_prompt, run, call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockContext:
    """Minimal StageContext mock for testing."""
    def __init__(self, state=None):
        self._state = state or {}
        self._logger = logging.getLogger("SpecForge_Test")

    def get(self, key):
        return self._state.get(key)

    def set(self, key, value):
        self._state[key] = value

    @property
    def logger(self):
        return self._logger


# ---------------------------------------------------------------------------
# validate_spec tests
# ---------------------------------------------------------------------------

def test_validate_spec_valid():
    """Happy path — all required + optional fields, correct types."""
    print("Testing validate_spec with valid input...")
    spec = {
        "objective": "Build a calculator",
        "inputs": ["two numbers", "an operator"],
        "outputs": ["a result"],
        "functional_requirements": ["must add", "must subtract"],
        "non_functional_requirements": ["must be fast"],
        "constraints": ["no external dependencies"],
    }
    result = validate_spec(spec)
    assert result == spec, f"Expected spec unchanged, got {result}"
    print("  PASS")
    return True


def test_validate_spec_optional_defaults():
    """Missing optional fields should default to empty lists."""
    print("Testing validate_spec optional defaults...")
    spec = {
        "objective": "Build something",
        "inputs": ["data"],
        "outputs": ["result"],
        "functional_requirements": ["must work"],
    }
    result = validate_spec(spec)
    assert result["non_functional_requirements"] == [], \
        f"Expected [], got {result['non_functional_requirements']}"
    assert result["constraints"] == [], \
        f"Expected [], got {result['constraints']}"
    print("  PASS")
    return True


def test_validate_spec_missing_required():
    """Missing required key should raise ValueError."""
    print("Testing validate_spec missing required key...")
    spec = {
        "inputs": ["data"],
        "outputs": ["result"],
        "functional_requirements": ["must work"],
    }
    try:
        validate_spec(spec)
        print("  FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        assert "objective" in str(e), f"Error should mention 'objective': {e}"
        print("  PASS")
        return True


def test_validate_spec_extraneous_keys():
    """Extra keys should raise ValueError."""
    print("Testing validate_spec extraneous keys...")
    spec = {
        "objective": "Build something",
        "inputs": ["data"],
        "outputs": ["result"],
        "functional_requirements": ["must work"],
        "description": "This should not be here",
    }
    try:
        validate_spec(spec)
        print("  FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        assert "description" in str(e), f"Error should mention 'description': {e}"
        print("  PASS")
        return True


def test_validate_spec_wrong_type_objective():
    """objective must be str, not list."""
    print("Testing validate_spec wrong type for objective...")
    spec = {
        "objective": ["should be a string"],
        "inputs": ["data"],
        "outputs": ["result"],
        "functional_requirements": ["must work"],
    }
    try:
        validate_spec(spec)
        print("  FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        assert "objective" in str(e), f"Error should mention 'objective': {e}"
        print("  PASS")
        return True


def test_validate_spec_wrong_type_list_items():
    """List fields must contain only strings."""
    print("Testing validate_spec wrong type in list items...")
    spec = {
        "objective": "Build something",
        "inputs": [123, "data"],
        "outputs": ["result"],
        "functional_requirements": ["must work"],
    }
    try:
        validate_spec(spec)
        print("  FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        assert "inputs" in str(e), f"Error should mention 'inputs': {e}"
        print("  PASS")
        return True


def test_validate_spec_empty_required_list():
    """Required list fields must not be empty."""
    print("Testing validate_spec empty required list...")
    spec = {
        "objective": "Build something",
        "inputs": [],
        "outputs": ["result"],
        "functional_requirements": ["must work"],
    }
    try:
        validate_spec(spec)
        print("  FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        assert "inputs" in str(e), f"Error should mention 'inputs': {e}"
        print("  PASS")
        return True


def test_build_prompt_contains_idea():
    """Prompt must include the raw idea verbatim."""
    print("Testing build_prompt contains idea...")
    idea = "Build a REST API that calculates averages."
    prompt = build_prompt(idea)
    assert idea in prompt, f"Idea not found in prompt: {prompt}"
    print("  PASS")
    return True


def test_run_rejects_empty_idea():
    """run() should return False when idea_raw is None or empty."""
    print("Testing run() rejects empty idea...")
    ctx_none = MockContext({"idea_raw": None})
    ctx_empty = MockContext({"idea_raw": ""})
    ctx_missing = MockContext({})

    assert run(ctx_none) is False, "Should reject None idea"
    assert run(ctx_empty) is False, "Should reject empty idea"
    assert run(ctx_missing) is False, "Should reject missing idea"
    print("  PASS")
    return True


def test_run_hardened_json_object_enforcement():
    """run() should reject JSON arrays even in JSON mode."""
    print("Testing run() enforcement of JSON object (dict)...")
    
    with patch("Stage_1.call_llm", return_value="[1, 2, 3]"):
        ctx = MockContext({"idea_raw": "valid idea"})
        assert run(ctx) is False, "Should reject JSON array"
    
    print("  PASS")
    return True


def test_run_robust_markdown_regex():
    """run() should strip Markdown fences using the 2-step regex."""
    print("Testing run() robust markdown regex...")
    
    fenced_input = "```json\n{\"objective\": \"test\", \"inputs\": [\"i\"], \"outputs\": [\"o\"], \"functional_requirements\": [\"r\"]}\n```"
    
    with patch("Stage_1.call_llm", return_value=fenced_input):
        ctx = MockContext({"idea_raw": "valid idea"})
        assert run(ctx) is True, "Should strip fences and succeed"
        assert ctx.get("spec")["objective"] == "test"
    
    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all Stage 1 tests."""
    print("Running Stage 1 Specification Engine Tests\n")

    tests = [
        test_validate_spec_valid,
        test_validate_spec_optional_defaults,
        test_validate_spec_missing_required,
        test_validate_spec_extraneous_keys,
        test_validate_spec_wrong_type_objective,
        test_validate_spec_wrong_type_list_items,
        test_validate_spec_empty_required_list,
        test_build_prompt_contains_idea,
        test_run_rejects_empty_idea,
        test_run_hardened_json_object_enforcement,
        test_run_robust_markdown_regex,
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

    print(f"\nTest Results: {passed}/{total} tests passed")

    if passed == total:
        print("All Stage 1 tests passed!")
        return True
    else:
        print("Some tests need attention.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
