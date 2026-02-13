"""
SpecForge Stage 3 — Structural Decomposition

Defines the structural interface descriptors (exports) for architectural 
components that are depended upon. 

Refocused on instruction design:
- Softened: Removes AST/Regex code enforcement.
- Hardened: Strictly enforces JSON schema and minimality.
- Literalism: Forbids introducing any detail not present in Stage 1/2.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Set

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """\
You are a structural decomposition engine. Your sole job is to define the \
minimal interface descriptors (exports) for architectural components.

Rules for Decomposition:
1. Output MUST be a strictly valid JSON object with exactly one key: "files".
2. "files" MUST be a list of objects, each containing:
   - "path": Must match a path from the architecture exactly.
   - "exports": A list of objects, each containing:
     - "name": Descriptive name of the exported function/interface.
     - "inputs": A list of strings describing inputs (e.g., "id: int").
     - "returns": A string describing the return type (e.g., "bool").
3. PARITY: You MUST define exports for EVERY file that is listed as a dependency in the architecture.
4. MINIMALITY: Only define exports for files that have dependents.
5. ANTI-CREEP: Do not introduce abstraction layers, "helper" interface fields, \
or implied methods. If it's not strictly necessary for the architectural \
intent, exclude it.
6. LITERALISM: If a detail is not present in the original spec or derived \
architecture, it MUST NOT be introduced. Choose the simplest possible interface.
7. ABSOLUTELY FORBIDDEN:
   - Implementation logic or code snippets.
   - Class definitions or decorators.
   - Framework or technology assumptions.
8. Your output must be ONLY the raw JSON object. Do not provide any conversational text.\
"""

# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------

def validate_structural_decomposition(raw: dict, depended_paths: set) -> dict:
    """
    Validate the structural decomposition JSON.
    Ensures schema correctness and strict alignment with architecture.
    """
    if set(raw.keys()) != {"files"}:
        raise ValueError("Root must contain exactly one key: 'files'")
    
    files = raw["files"]
    if not isinstance(files, list):
        raise ValueError("'files' must be a list")

    contract_paths = set()
    for i, entry in enumerate(files):
        if not isinstance(entry, dict) or set(entry.keys()) != {"path", "exports"}:
            raise ValueError(f"Entry {i} must have exactly 'path' and 'exports'")
        
        path = entry["path"]
        exports = entry["exports"]

        if path not in depended_paths:
            # logger.warning(f"File listed in decomposition but not depended upon: {path}")
            pass
        
        if path in contract_paths:
            raise ValueError(f"Duplicate path: {path}")
        contract_paths.add(path)

        if not isinstance(exports, list) or len(exports) == 0:
            raise ValueError(f"File '{path}' must have at least one export")

        seen_names = set()
        for j, exp in enumerate(exports):
            if not isinstance(exp, dict) or set(exp.keys()) != {"name", "inputs", "returns"}:
                raise ValueError(f"File '{path}': Export {j} must have 'name', 'inputs', 'returns'")
            
            name = exp["name"]
            if name in seen_names:
                raise ValueError(f"Duplicate export name '{name}' in '{path}'")
            seen_names.add(name)

    # Parity check
    missing = depended_paths - contract_paths
    if missing:
        raise ValueError(f"Missing interface descriptors for depended-upon files: {missing}")

    return raw

# ---------------------------------------------------------------------------
# Provider Interactions
# ---------------------------------------------------------------------------

async def _run_backboard(context, user_prompt: str, model: str, depended_paths: set):
    from backboard import BackboardClient
    logger = context.logger
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 3: BACKBOARD_API_KEY not set")
        return False

    async with BackboardClient(api_key=api_key, timeout=60) as client:
        assistant = await client.create_assistant(
            name="SpecForge_Stage3",
            system_prompt=SYSTEM_PROMPT,
        )
        try:
            thread = await client.create_thread(assistant_id=assistant.assistant_id)
            current_user_prompt = user_prompt
            
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 3: Attempt {attempt}/{MAX_ATTEMPTS}")
                try:
                    response = await client.add_message(
                        thread_id=thread.thread_id,
                        content=current_user_prompt,
                        llm_provider="openai",
                        model_name=model,
                    )
                    cleaned = response.content.strip()
                    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
                    cleaned = re.sub(r"\n```$", "", cleaned)
                    
                    parsed = json.loads(cleaned)
                    validated = validate_structural_decomposition(parsed, depended_paths)
                    context.set("contracts", validated)
                    logger.info("Stage 3: Structural decomposition validated")
                    return True
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Stage 3: Attempt {attempt} failed: {e}")
                    current_user_prompt = f"The previous attempt failed validation with the following error:\n{e}\n\nPlease fix this and regenerate the JSON correctly."
        finally:
            await client.delete_assistant(assistant.assistant_id)
    return False

def _run_openai_synchronous(context, user_prompt: str, model: str, depended_paths: set) -> bool:
    from openai import OpenAI
    logger = context.logger
    client = OpenAI()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Stage 3: Attempt {attempt}/{MAX_ATTEMPTS}")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            validated = validate_structural_decomposition(parsed, depended_paths)
            context.set("contracts", validated)
            logger.info("Stage 3: Structural decomposition validated")
            return True
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Stage 3: Attempt {attempt} failed: {e}")
            messages.append({"role": "assistant", "content": content if 'content' in locals() else "{}"})
            messages.append({"role": "user", "content": f"Validation failed: {e}. Please fix and regenerate."})
    return False

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run(context) -> bool:
    logger = context.logger
    logger.info("Stage 3: Structural Decomposition started")

    arch = context.get("architecture")
    if not isinstance(arch, dict) or "files" not in arch:
        logger.error("Stage 3: Architecture missing or malformed")
        return False
    
    depended_paths = set()
    for f in arch["files"]:
        for d in f.get("depends_on", []):
            depended_paths.add(d)
    
    user_prompt = f"Architecture:\n{json.dumps(arch, indent=2)}\n\nGenerate the structural decomposition."
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini")

    if provider == "backboard":
        import asyncio
        return asyncio.run(_run_backboard(context, user_prompt, model, depended_paths))
    else:
        return _run_openai_synchronous(context, user_prompt, model, depended_paths)
