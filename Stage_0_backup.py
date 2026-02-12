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
from typing import Dict, Any, Optional, Callable
from pathlib import Path


class StateManager:
    """Centralized state management with controlled access interfaces."""
    
    def __init__(self):
        self._state = {
            "idea_raw": None,
            "spec": None,
            "architecture": None,
            "contracts": None,
            "generated_files": {},
            "validation_results": {},
            "stage_status": {}
        }
    
    def get(self, key: str) -> Any:
        """Get value from state."""
        return self._state.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._state[key] = value
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete state copy."""
        return self._state.copy()
    
    def update_stage_status(self, stage: int, status: str) -> None:
        """Update stage execution status."""
        self._state["stage_status"][stage] = status


class InputHandler:
    """Handles raw user input acceptance without interpretation."""
    
    @staticmethod
    def accept_user_input() -> str:
        """Accept and return raw user input exactly as provided."""
        print("=== SpecForge Stage 0: System Initialization ===")
        print("Please describe your project idea:")
        print("(Input will be stored exactly as provided without modification)")
        print()
        
        # Read raw input preserving all formatting
        user_input = sys.stdin.read().strip()
        
        if not user_input:
            print("No input provided. Using interactive prompt...")
            user_input = input("Project description: ")
        
        return user_input


class ConfigurationManager:
    """Manages deterministic system configuration."""
    
    def __init__(self):
        self.execution_order = [1, 2, 3, 4, 5]
        self.max_regeneration_attempts = 3
        self.deterministic_mode = True
        self.logging_enabled = True
        self.dry_run_mode = False
        
        # Stage registry - will be populated when stage modules are available
        self.STAGES = {
            1: None,  # stage1_spec.run
            2: None,  # stage2_architecture.run
            3: None,  # stage3_contracts.run
            4: None,  # stage4_generator.run
            5: None   # stage5_validator.run
        }
    
    def get_execution_order(self) -> list:
        """Get deterministic execution order."""
        return self.execution_order.copy()
    
    def register_stage(self, stage_num: int, stage_function: Callable) -> None:
        """Register a stage function in the registry."""
        if stage_num in self.STAGES:
            self.STAGES[stage_num] = stage_function
        else:
            raise ValueError(f"Invalid stage number: {stage_num}")


class StorageManager:
    """Manages controlled storage environment."""
    
    def __init__(self, base_dir: str = "builds"):
        self.base_dir = Path(base_dir)
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
        
    def initialize_logger(self) -> logging.Logger:
        """Initialize system logger with file and console handlers."""
        # Create logger
        self.logger = logging.getLogger("SpecForge")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
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
                import json
                with open(state_file, 'w') as f:
                    json.dump(state.get_state(), f, indent=2)
                self.logger.logger.info(f"Failure state preserved to {state_file}")
            except Exception as preserve_error:
                self.logger.log_error(preserve_error, "preserving failure state")


class StageOrchestrator:
    """Main orchestrator for stage execution with isolation enforcement."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.config_manager = ConfigurationManager()
        self.storage_manager = StorageManager()
        self.logging_system = LoggingSystem(self.storage_manager)
        self.error_boundary = ErrorBoundary(self.logging_system)
        self.logger = None
        
    def initialize_system(self) -> None:
        """Initialize all system components."""
        # Initialize storage first
        run_dir = self.storage_manager.initialize_storage()
        
        # Initialize logging
        self.logger = self.logging_system.initialize_logger()
        
        self.logger.info("SpecForge Stage 0: System initialization complete")
        self.logger.info(f"Run directory: {run_dir}")
    
    def accept_user_idea(self) -> None:
        """Accept and store raw user input."""
        raw_input = self.error_boundary.execute_with_boundary(
            InputHandler.accept_user_input
        )
        
        self.state_manager.set("idea_raw", raw_input)
        self.logger.info(f"User idea accepted and stored (length: {len(raw_input)} chars)")
    
    def register_stages(self) -> None:
        """Register stage functions (placeholder for future integration)."""
        # This will be populated when stage modules are implemented
        # For now, we create placeholder functions
        def placeholder_stage(state_manager, config):
            self.logger.info(f"Placeholder stage executed")
            return True
            
        for stage_num in self.config_manager.execution_order:
            self.config_manager.register_stage(stage_num, placeholder_stage)
    
    def execute_pipeline(self) -> bool:
        """Execute the complete pipeline with stage isolation."""
        self.logger.info("Starting pipeline execution")
        
        try:
            for stage_num in self.config_manager.get_execution_order():
                stage_function = self.config_manager.STAGES.get(stage_num)
                
                if not stage_function:
                    self.logger.error(f"Stage {stage_num} not registered")
                    return False
                
                # Update stage status
                self.state_manager.update_stage_status(stage_num, "running")
                self.logging_system.log_stage_start(stage_num)
                
                # Execute stage with controlled interface
                try:
                    success = self.error_boundary.execute_with_boundary(
                        stage_function,
                        self.state_manager,
                        self.config_manager
                    )
                    
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
        # Initialize system components
        orchestrator.initialize_system()
        
        # Accept user input
        orchestrator.accept_user_idea()
        
        # Register stages
        orchestrator.register_stages()
        
        # Execute pipeline (in dry run mode for Stage 0)
        orchestrator.config_manager.dry_run_mode = True
        
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