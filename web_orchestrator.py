import os
import sys
import json
import uuid
import logging
from Stage_0 import StageOrchestrator

def map_contracts(raw_contracts):
    """Map Stage 3 exports to frontend Contracts type."""
    mapped = {}
    if not raw_contracts or "files" not in raw_contracts:
        return mapped
    
    for file_entry in raw_contracts["files"]:
        path = file_entry["path"]
        functions = []
        for export in file_entry["exports"]:
            params = []
            for inp in export.get("inputs", []):
                if ":" in inp:
                    name, ptype = inp.split(":", 1)
                    params.append({"name": name.strip(), "type": ptype.strip()})
                else:
                    params.append({"name": inp.strip(), "type": "any"})
            
            functions.append({
                "name": export["name"],
                "parameters": params,
                "returnType": export.get("returns", "void")
            })
        
        mapped[path] = {"functions": functions}
    return mapped

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No idea provided"}))
        sys.exit(1)
    
    idea = sys.argv[1]
    project_id = str(uuid.uuid4())[:8]
    
    # Setup Orchestrator
    orchestrator = StageOrchestrator()
    orchestrator.register_stages()

    # Force Backboard provider
    os.environ["SPECFORGE_PROVIDER"] = "backboard"
    os.environ["BACKBOARD_API_KEY"] = "espr_-xJfRq2-EDhWYk0SvlNpZJjvZ-HJOU9HB5VriVdz1OY"
    os.environ["SPECFORGE_MODEL"] = "gpt-4o-mini"
    
    # Initialize storage and logger
    orchestrator.storage_manager.initialize_storage()
    orchestrator.logger = orchestrator.logging_system.initialize_logger()
    
    # Disable dry run
    orchestrator.config_manager.dry_run_mode = False
    
    # Set idea
    orchestrator.state_manager.set("idea_raw", idea)
    
    # Execute
    success = orchestrator.execute_pipeline()
    
    if not success:
        print(json.dumps({"error": "Pipeline execution failed", "logs": "Check system logs"}))
        sys.exit(1)
    
    state = orchestrator.state_manager
    spec = state.get("spec")
    arch = state.get("architecture")
    raw_contracts = state.get("contracts")
    prompt_pack_data = state.get("prompt_pack") or {}
    prompt_pack = prompt_pack_data.get("prompt_pack", [])
    
    # Construct final project object
    project = {
        "id": project_id,
        "spec": {
            "project_name": spec.get("project_name", "Untitled Project"),
            "core_objective": spec.get("core_objective", ""),
            "functional_requirements": spec.get("functional_requirements", []),
            "constraints": spec.get("constraints", []),
            "required_libraries": spec.get("required_libraries", [])
        },
        "architecture": {
            "files": [{"path": f["path"]} for f in arch.get("files", [])]
        },
        "contracts": map_contracts(raw_contracts),
        "prompt_pack": prompt_pack
    }
    
    # Store in builds/web_projects/ for persistence
    storage_path = os.path.join("builds", "web_projects", f"{project_id}.json")
    with open(storage_path, "w") as f:
        json.dump(project, f, indent=2)
    
    # Output only the project JSON to stdout
    print(json.dumps(project))

if __name__ == "__main__":
    main()
