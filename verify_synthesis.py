"""
End-to-end verification for Prompt Synthesis Pipeline.
Runs Stage 1 -> 2 -> 3 -> 4 -> 5.
"""

import os
import json
from Stage_0 import StageOrchestrator

def main():
    # Set environment for Backboard verification
    os.environ["SPECFORGE_PROVIDER"] = "backboard"
    os.environ["BACKBOARD_API_KEY"] = "espr_-xJfRq2-EDhWYk0SvlNpZJjvZ-HJOU9HB5VriVdz1OY"
    os.environ["SPECFORGE_MODEL"] = "gpt-4o"
    
    orchestrator = StageOrchestrator()
    orchestrator.register_stages()
    
    # Initialize storage and logger
    orchestrator.storage_manager.initialize_storage()
    orchestrator.logger = orchestrator.logging_system.initialize_logger()
    
    # Store the idea
    idea = "Create a random space haiku generator."
    orchestrator.state_manager.set("idea_raw", idea)
    
    # Execute pipeline
    print("\n--- Executing Stage 1 -> 2 -> 3 -> 4 ---\n")
    success = orchestrator.execute_pipeline()
    
    # Analyze results
    state = orchestrator.state_manager
    s1_success = state.get("spec") is not None
    s2_success = state.get("architecture") is not None
    s3_success = state.get("contracts") is not None
    
    pack_data = state.get("prompt_pack")
    prompt_pack = pack_data.get("prompt_pack", []) if pack_data else []
    s4_success = len(prompt_pack) > 0
    
    print("\n=== Pipeline Result ===")
    print(f"Overall Success: {success}")
    print(f"Stage 1 Success: {s1_success}")
    print(f"Stage 2 Success: {s2_success}")
    print(f"Stage 3 Success: {s3_success}")
    print(f"Stage 4 Success: {s4_success}")
    
    if s4_success:
        print("\n--- PROMPT PACK ---")
        for entry in prompt_pack:
            path = entry["path"]
            prompt = entry["implementation_prompt"]
            print(f"\nFILE: {path}")
            print("-" * 20)
            print(prompt)
            print("-" * 20)
    else:
        print("\nPrompt Synthesis Failed or was skipped.")

if __name__ == "__main__":
    main()
