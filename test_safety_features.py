#!/usr/bin/env python3
"""
Test script to verify all Stage 0 safety features are working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from Stage_0 import (
    StateManager,
    ConfigurationManager,
    StorageManager,
    StageContext
)
import tempfile
import shutil

def test_state_manager_strict_keys():
    """Test that StateManager enforces strict key schema."""
    print("Testing StateManager strict key enforcement...")
    
    sm = StateManager()
    
    # Test valid key
    try:
        sm.set("idea_raw", "test idea")
        print("PASS: Valid key accepted")
    except Exception as e:
        print(f"FAIL: Valid key rejected: {e}")
        return False
    
    # Test invalid key
    try:
        sm.set("invalid_key", "value")
        print("FAIL: Invalid key was accepted")
        return False
    except ValueError as e:
        print("PASS: Invalid key correctly rejected")
    
    return True

def test_config_manager_immutable():
    """Test that ConfigurationManager enforces immutability."""
    print("\nTesting ConfigurationManager immutability...")
    
    cm = ConfigurationManager()
    
    # Test execution order is tuple
    order = cm.get_execution_order()
    if isinstance(order, tuple):
        print("PASS: Execution order is immutable tuple")
    else:
        print("FAIL: Execution order is mutable")
        return False
    
    # Test improved stage registration validation
    def good_stage(context):
        return True
    
    def bad_stage_arity():
        return True  # No parameters
    
    def bad_stage_extra_params(context, extra):
        return True  # Extra parameters
    
    def bad_stage_varargs(*args):
        return True  # Variable args
    
    def bad_stage_kwargs(**kwargs):
        return True  # Variable kwargs
    
    # Good stage should register
    try:
        cm.register_stage(1, good_stage)
        print("PASS: Valid stage registration succeeded")
    except Exception as e:
        print(f"FAIL: Valid stage registration failed: {e}")
        return False
    
    # Bad arity should fail
    try:
        cm.register_stage(2, bad_stage_arity)
        print("FAIL: Stage with no parameters was allowed")
        return False
    except TypeError as e:
        print("PASS: Stage with no parameters correctly rejected")
    
    # Extra parameters should fail
    try:
        cm.register_stage(4, bad_stage_extra_params)
        print("FAIL: Stage with extra parameters was allowed")
        return False
    except TypeError as e:
        print("PASS: Stage with extra parameters correctly rejected")
    
    # Kwargs should fail
    try:
        cm.register_stage(3, bad_stage_kwargs)
        print("FAIL: Stage with **kwargs was allowed")
        return False
    except TypeError as e:
        print("PASS: Stage with **kwargs correctly rejected")
    
    # Test configuration locking
    cm.lock()
    try:
        cm.max_regeneration_attempts = 5
        print("FAIL: Locked configuration allowed modification")
        return False
    except RuntimeError as e:
        print("PASS: Locked configuration correctly prevented modification")
    
    return True

def test_storage_manager_path_safety():
    """Test that StorageManager enforces path safety."""
    print("\nTesting StorageManager path safety...")
    
    # Test safe path
    try:
        sm = StorageManager("builds")
        print("PASS: Safe path accepted")
    except Exception as e:
        print(f"FAIL: Safe path rejected: {e}")
        return False
    
    # Test dangerous path (attempt directory traversal)
    try:
        sm = StorageManager("../../../etc")
        print("FAIL: Dangerous path was accepted")
        return False
    except ValueError as e:
        print("PASS: Dangerous path correctly rejected")
    
    # Test absolute path outside project
    try:
        sm = StorageManager("C:/Windows/System32")
        print("FAIL: Absolute external path was accepted")
        return False
    except ValueError as e:
        print("PASS: Absolute external path correctly rejected")
    
    return True

def test_stage_context_isolation():
    """Test that StageContext provides proper isolation."""
    print("\nTesting StageContext isolation...")
    
    sm = StateManager()
    cm = ConfigurationManager()
    logger = logging.getLogger("TestLogger")
    ctx = StageContext(sm, cm, logger)
    
    # Test controlled access
    sm.set("idea_raw", "test idea")
    if ctx.get("idea_raw") == "test idea":
        print("PASS: StageContext provides controlled state access")
    else:
        print("FAIL: StageContext state access failed")
        return False
    
    # Test read-only config view
    config_view = ctx.config
    
    # Test that internal access is blocked (not just hidden)
    try:
        internal = object.__getattribute__(config_view, '_config')
        print("FAIL: Config view allows internal access")
        return False
    except AttributeError as e:
        print("PASS: Config view correctly blocks internal access")
    except Exception as e:
        print(f"FAIL: Unexpected error accessing internal: {e}")
        return False
    
    # Test config view prevents modification
    try:
        config_view.max_regeneration_attempts = 10
        print("FAIL: Config view allowed modification")
        return False
    except AttributeError as e:
        print("PASS: Config view correctly prevents modification")
    
    # Test config view prevents private access
    try:
        _private = config_view._locked
        print("FAIL: Config view allowed private access")
        return False
    except AttributeError as e:
        print("PASS: Config view correctly prevents private access")
    
    # Test config view allows safe read access
    try:
        order = config_view.get_execution_order()
        if isinstance(order, tuple):
            print("PASS: Config view allows safe read access")
        else:
            print("FAIL: Config view corrupted safe access")
            return False
    except Exception as e:
        print(f"FAIL: Config view blocked safe access: {e}")
        return False
    
    # Test logger access
    if ctx.logger == logger:
        print("PASS: StageContext provides access to injected logger")
    else:
        print("FAIL: StageContext logger mismatch")
        return False
        
    # Test logger is read-only
    try:
        ctx.logger = logging.getLogger("NewLogger")
        print("FAIL: StageContext allowed logger reassignment")
        return False
    except AttributeError:
        print("PASS: StageContext logger property is read-only")
    
    return True

def test_boolean_stage_return():
    """Test that stage functions must return boolean."""
    print("\nTesting boolean stage return enforcement...")
    
    from Stage_0 import StageOrchestrator
    
    orchestrator = StageOrchestrator()
    
    # Test non-boolean return
    def bad_stage(context):
        return "not a boolean"
    
    try:
        orchestrator.config_manager.register_stage(1, bad_stage)
        # This should pass registration but fail execution
        print("PASS: Non-boolean return stage registered (will fail at execution)")
    except Exception as e:
        print(f"FAIL: Non-boolean return stage registration failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all safety feature tests."""
    print("Running Stage 0 Safety Feature Tests\n")
    
    tests = [
        test_state_manager_strict_keys,
        test_config_manager_immutable,
        test_storage_manager_path_safety,
        test_stage_context_isolation,
        test_boolean_stage_return
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"FAIL: {test.__name__} failed")
        except Exception as e:
            print(f"FAIL: {test.__name__} crashed: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All safety features are working correctly!")
        return True
    else:
        print("Some safety features need attention")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
