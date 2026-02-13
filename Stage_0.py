"""
SpecForge Stage 0 - System Initialization and Orchestration

Stage 0 is strictly orchestration and preparation.
It prepares the system, makes no decisions, and enforces deterministic execution.

Primary Responsibilities:
- Accept raw user input
- Initialize system state
- Define deterministic configuration
- Create stage execution registry
- Initialize storage environment
- Set up logging system
- Establish error boundaries
- Enforce stage isolation
"""

import os
import sys
import logging
import datetime
import copy
import json
import uuid
from typing import Dict, Any, Optional, Callable
from pathlib import Path


class StateManager:
    """Centralized state management with controlled access interfaces."""
    
    ALLOWED_KEYS = {
        "idea_raw",
        "spec",
        "architecture",
        "contracts",
        "prompt_pack",
        "validation_results",
        "stage_status"
    }
    
    def __init__(self):
        self._state = {
            "idea_raw": None,
            "spec": None,
            "architecture": None,
            "contracts": None,
            "prompt_pack": {},
            "validation_results": {},
            "stage_status": {}
        }
    
    def get(self, key: str) -> Any:
        """Get value from state."""
        return self._state.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        if key not in self.ALLOWED_KEYS:
            raise ValueError(f"Invalid state key: {key}")
        self._state[key] = value
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete state copy."""
        # Fast copy for simple state structure
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self._state.items()}
    
    def update_stage_status(self, stage: int, status: str) -> None:
        """Update stage execution status."""
        current_status = self.get("stage_status") or {}
        current_status[stage] = status
        self.set("stage_status", current_status)
    



class InputHandler:
    """Handles raw user input acceptance without interpretation."""
    
    @staticmethod
    def accept_user_input() -> str:
        """Accept and return raw user input exactly as provided."""
        print("=== SpecForge Stage 0: System Initialization ===")
        print("Please describe your project idea:")
        print("(Input will be stored exactly as provided without modification)")
        print()
        
        # Try to read from stdin first (for piped input)
        try:
            import select
            if hasattr(select, 'select'):  # Unix-like systems
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    user_input = sys.stdin.read().strip()
                else:
                    raise EOFError
            else:  # Windows fallback
                user_input = sys.stdin.read().strip()
        except (EOFError, OSError, AttributeError):
            # Fallback to interactive prompt or default
            try:
                user_input = input("Project description: ")
            except EOFError:
                # Default example for non-interactive environments
                user_input = "Build a REST API that calculates averages."
                print(f"No interactive input available. Using example: {user_input}")
        
        return user_input


class ConfigurationManager:
    """Manages deterministic system configuration."""
    
    def __init__(self):
        self._execution_order = (1, 2, 3, 4)
        self.max_regeneration_attempts = 3
        self.deterministic_mode = True
        self.logging_enabled = True
        self.dry_run_mode = False
        self._locked = False
        
        # Stage registry - will be populated when stage modules are available
        self.STAGES = {
            1: None,  # stage1_spec.run
            2: None,  # stage2_architecture.run
            3: None,  # stage3_contracts.run
            4: None   # stage4_generator.run
        }
    
    def get_execution_order(self) -> tuple:
        """Get deterministic execution order."""
        return self._execution_order
    
    def register_stage(self, stage_num: int, stage_function: Callable) -> None:
        """Register a stage function in the registry."""
        if stage_num not in self.STAGES:
            raise ValueError(f"Invalid stage number: {stage_num}")
        
        if self.STAGES[stage_num] is not None:
            raise RuntimeError(f"Stage {stage_num} already registered")
        
        if not callable(stage_function):
            raise TypeError("Stage function must be callable")
        
        if self._locked:
            raise RuntimeError("Configuration is locked")
            
        self.STAGES[stage_num] = stage_function
    
    def lock(self) -> None:
        """Lock configuration to prevent modifications."""
        self._locked = True
    
    def __setattr__(self, key, value):
        if getattr(self, "_locked", False) and key != '_locked':
            raise RuntimeError("Configuration is locked")
        super().__setattr__(key, value)


class StorageManager:
    """Manages controlled storage environment."""
    
    def __init__(self, base_dir: str = "builds"):
        # Resolve to absolute path for reliable comparison
        self.base_dir = Path(base_dir).resolve()
        project_root = Path.cwd().resolve()
        
        # Strict prefix check - must be within project directory
        try:
            self.base_dir.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Storage base_dir must be inside project directory: {self.base_dir} not under {project_root}")
        
        self.current_run_dir = None
        
    def initialize_storage(self) -> str:
        """Create isolated project namespace for current run."""
        # Create base builds directory if missing
        self.base_dir.mkdir(exist_ok=True)
        
        # Generate unique run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        self.current_run_dir = self.base_dir / run_id
        
        # Create run directory
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.current_run_dir / "output").mkdir(exist_ok=True)
        (self.current_run_dir / "logs").mkdir(exist_ok=True)
        (self.current_run_dir / "state").mkdir(exist_ok=True)
        
        return str(self.current_run_dir)
    
    def get_run_directory(self) -> Optional[str]:
        """Get current run directory path."""
        return str(self.current_run_dir) if self.current_run_dir else None


class LoggingSystem:
    """Manages structured logging for the system."""
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        self.logger = None
        self._run_id = None
        
    def initialize_logger(self) -> logging.Logger:
        """Initialize system logger with file and console handlers."""
        # Generate run-scoped logger name with UUID to avoid collisions
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        logger_name = f"SpecForge_Run_{run_id}_{unique_id}"
        
        # Create/get run-scoped logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Only configure if not already configured for this run
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            # File handler (if storage is initialized)
            if self.storage_manager.get_run_directory():
                log_file = Path(self.storage_manager.get_run_directory()) / "logs" / "system.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            
            self.logger.addHandler(console_handler)
        
        self.logger.info("SpecForge Stage 0: Logging system initialized")
        return self.logger
    
    def log_stage_start(self, stage: int) -> None:
        """Log stage execution start."""
        if self.logger:
            self.logger.info(f"Starting Stage {stage}")
    
    def log_stage_complete(self, stage: int) -> None:
        """Log stage execution completion."""
        if self.logger:
            self.logger.info(f"Completed Stage {stage}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with context."""
        if self.logger:
            self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)


class ErrorBoundary:
    """Provides controlled error handling and system boundaries."""
    
    def __init__(self, logger: LoggingSystem):
        self.logger = logger
        
    def execute_with_boundary(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function within controlled error boundary."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.log_error(e, f"executing {func.__name__}")
            raise  # Re-raise to stop execution on critical failure
    
    def handle_critical_failure(self, error: Exception, state: StateManager) -> None:
        """Handle critical system failure with state preservation."""
        self.logger.log_error(error, "critical system failure")
        
        # Preserve partial state for debugging
        if self.logger.storage_manager.get_run_directory():
            state_file = Path(self.logger.storage_manager.get_run_directory()) / "state" / "failure_state.json"
            try:
                # Safe JSON serialization
                def safe_serialize(obj):
                    try:
                        json.dumps(obj)
                        return obj
                    except TypeError:
                        return str(obj)
                
                safe_state = {k: safe_serialize(v) for k, v in state.get_state().items()}
                
                with open(state_file, 'w') as f:
                    json.dump(safe_state, f, indent=2)
                self.logger.logger.info(f"Failure state preserved to {state_file}")
            except Exception as preserve_error:
                self.logger.log_error(preserve_error, "preserving failure state")


class ReadOnlyConfigView:
    """Provides read-only access to configuration."""
    
    def __init__(self, config_manager: ConfigurationManager):
        object.__setattr__(self, '_ReadOnlyConfigView__config', config_manager)
    
    def __getattr__(self, name: str) -> Any:
        # Block all private attribute access
        if name.startswith('_'):
            raise AttributeError("Cannot access private attributes")
        
        config = object.__getattribute__(self, '_ReadOnlyConfigView__config')
        value = getattr(config, name)
        
        # Only allow safe methods
        if callable(value) and name not in ['get_execution_order']:
            raise AttributeError(f"Cannot access method '{name}' on read-only config view")
        
        return value
    
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Cannot modify read-only config view")
    
    def __delattr__(self, name: str) -> None:
        raise AttributeError("Cannot delete from read-only config view")


class StageContext:
    """Provides controlled interface for stages to access state and configuration."""
    
    def __init__(self, state_manager: StateManager, config_manager: ConfigurationManager, logger: logging.Logger, run_directory: str = ""):
        self._state_manager = state_manager
        self._config_view = ReadOnlyConfigView(config_manager)
        self._logger = logger
        self._run_directory = run_directory
        
    def get(self, key: str) -> Any:
        """Get value from state."""
        return self._state_manager.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._state_manager.set(key, value)
    
    @property
    def config(self) -> ReadOnlyConfigView:
        """Get read-only configuration view."""
        return self._config_view

    @property
    def run_directory(self) -> str:
        """Get the current run directory."""
        return self._run_directory

    @property
    def logger(self) -> logging.Logger:
        """Get the active run-scoped logger."""
        return self._logger


class StageOrchestrator:
    """Main orchestrator for stage execution with isolation enforcement."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.config_manager = ConfigurationManager()
        self.storage_manager = StorageManager()
        self.logging_system = LoggingSystem(self.storage_manager)
        self.error_boundary = ErrorBoundary(self.logging_system)
        self.logger = None
        
        
    def accept_user_idea(self) -> None:
        """Accept and store raw user input."""
        raw_input = self.error_boundary.execute_with_boundary(
            InputHandler.accept_user_input
        )
        
        self.state_manager.set("idea_raw", raw_input)
        self.logger.info(f"User idea accepted and stored (length: {len(raw_input)} chars)")
    
    def register_stages(self) -> None:
        """Register stage placeholders (lazy loaded at runtime)."""
        # We store the module names/mapping instead of importing them here
        for i in range(1, 5):
            self.config_manager.STAGES[i] = f"Stage_{i}"

    def execute_pipeline(self) -> bool:
        """Execute the complete pipeline with stage isolation and lazy loading."""
        self.logger.info("Starting pipeline execution")
        
        if self.config_manager.dry_run_mode:
            self.logger.info("Dry run mode active — skipping execution")
            return True
        
        try:
            run_dir = self.storage_manager.get_run_directory() or ""
            context = StageContext(self.state_manager, self.config_manager, self.logger, run_dir)
            import importlib
            
            for stage_num in self.config_manager.get_execution_order():
                stage_target = self.config_manager.STAGES.get(stage_num)
                if not stage_target:
                    self.logger.error(f"Stage {stage_num} not registered")
                    return False
                
                # Lazy load stage function
                try:
                    module = importlib.import_module(stage_target)
                    stage_function = getattr(module, 'run')
                except (ImportError, AttributeError) as e:
                    self.logger.error(f"Failed to load {stage_target}: {e}")
                    return False

                # Update stage status
                self.state_manager.update_stage_status(stage_num, "running")
                self.logging_system.log_stage_start(stage_num)
                
                # Execute stage with controlled interface
                try:
                    success = self.error_boundary.execute_with_boundary(
                        stage_function,
                        context
                    )
                    
                    # Enforce boolean stage return
                    if not isinstance(success, bool):
                        raise TypeError(f"Stage {stage_num} must return boolean, got {type(success)}")
                    
                    if success:
                        self.state_manager.update_stage_status(stage_num, "completed")
                        self.logging_system.log_stage_complete(stage_num)
                    else:
                        self.state_manager.update_stage_status(stage_num, "failed")
                        self.logger.error(f"Stage {stage_num} returned failure")
                        return False
                        
                except Exception as e:
                    self.state_manager.update_stage_status(stage_num, "error")
                    self.error_boundary.handle_critical_failure(e, self.state_manager)
                    return False
            
            self.logger.info("Pipeline execution completed successfully")
            return True
            
        except Exception as e:
            self.error_boundary.handle_critical_failure(e, self.state_manager)
            return False
    
    def handoff_to_stage_1(self) -> Dict[str, Any]:
        """Provide controlled handoff to Stage 1."""
        self.logger.info("Preparing handoff to Stage 1")
        
        handoff_data = {
            "state": self.state_manager.get_state(),
            "config": {
                "deterministic_mode": self.config_manager.deterministic_mode,
                "max_regeneration_attempts": self.config_manager.max_regeneration_attempts
            },
            "run_directory": self.storage_manager.get_run_directory()
        }
        
        self.logger.info("Handoff to Stage 1 prepared")
        return handoff_data


def run() -> Dict[str, Any]:
    """
    Main entry point for Stage 0.
    
    Returns:
        Dict containing handoff data for Stage 1
    """
    orchestrator = StageOrchestrator()
    
    try:
        # Initialize storage first (before logging)
        run_dir = orchestrator.storage_manager.initialize_storage()
        
        # Initialize logging
        orchestrator.logger = orchestrator.logging_system.initialize_logger()
        
        # Accept user input BEFORE freezing state
        orchestrator.accept_user_idea()
        
        # Register stages BEFORE locking config
        orchestrator.register_stages()
        
        # Set dry_run_mode BEFORE locking config
        orchestrator.config_manager.dry_run_mode = True
        
        # Lock config to prevent runtime drift
        orchestrator.config_manager.lock()
        
        orchestrator.logger.info("SpecForge Stage 0: System initialization complete")
        orchestrator.logger.info(f"Run directory: {run_dir}")
        
        # Note: Actual pipeline execution would happen here
        # For Stage 0, we prepare the system but don't execute full pipeline
        orchestrator.logger.info("Stage 0 preparation complete - ready for Stage 1")
        
        # Provide handoff to Stage 1
        return orchestrator.handoff_to_stage_1()
        
    except Exception as e:
        if orchestrator.logger:
            orchestrator.error_boundary.handle_critical_failure(e, orchestrator.state_manager)
        else:
            print(f"Critical failure during Stage 0 initialization: {e}")
        raise


if __name__ == "__main__":
    # Run Stage 0 when executed directly
    handoff_data = run()
    print("\n=== Stage 0 Complete ===")
    print(f"Run directory: {handoff_data['run_directory']}")
    print(f"User idea stored: {bool(handoff_data['state']['idea_raw'])}")
    print("System ready for Stage 1 handoff")
