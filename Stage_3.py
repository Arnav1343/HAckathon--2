"""
SpecForge Stage 3 — Interface Contracts

Defines the public surface (exports) of files that are depended upon by other files.
Enforces strict interface freezing without implementation leakage.

Responsibilities:
- Read 'architecture' from state and fail if missing/malformed.
- Identify files with dependents; these must have exports.
- Enforce strict JSON schema: {"files": [{"path": str, "exports": [{"name": str, "inputs": list[str], "returns": str}]}]}.
- Exact match: The set of files in contracts must match the set of depended-upon files.
- Enforce regex-hardened signatures:
    - Name: ^[a-zA-Z_][a-zA-Z0-9_]*$
    - Input: ^[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*[a-zA-Z0-9_\[\]]+$
    - Returns: ^[a-zA-Z0-9_\[\]]+$
- Ensure len(exports) >= 1 for every entry.
- Ensure unique export names per file.
- Retry up to MAX_ATTEMPTS on violation.
- Write validated contracts to state.

Design Principles:
- Interface surface only.
- Zero implementation leakage (no classes, logic, or frameworks).
- No graph mutation (Stage 3 cannot add/rename files or change deps).
- Bounded retries (Reject → Retry → Fail).
"""

import os
import re
import json
import logging

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """\
You are an interface contract engine. Your sole job is to define the public \
surface (exports) for architectural components.

Rules for Contracts:
1. Output MUST be a strictly valid JSON object with exactly one key: "files".
2. "files" MUST be a list of objects, each containing:
   - "path": Must match a path from the architecture exactly.
   - "exports": A list of objects, each containing:
     - "name": Function name (snake_case).
     - "inputs": A list of strings in "name: type" format (e.g., "id: int").
     - "returns": A simple type string (e.g., "bool", "list[str]").
3. EXPORT MINIMALITY: Define exports ONLY for files that have dependents.
4. ABSOLUTELY FORBIDDEN:
   - Implementation logic or code snippets.
   - Class definitions or decorators.
   - Framework or technology assumptions.
   - Internal helper functions.
   - Complex or cross-module type definitions.
   - Modifying the dependency graph or adding new files.
5. Every file listed MUST have at least one export.

Your output must be ONLY the raw JSON object. Do not provide any conversational text.\
"""

# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------

NAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
INPUT_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*[a-zA-Z0-9_\[\]]+$")
RETURN_REGEX = re.compile(r"^[a-zA-Z0-9_\[\]]+$")

def validate_contracts(raw: dict, depended_paths: set) -> dict:
    """
    Strictly validate the interface contracts against architecture and schema.
    Returns the validated dict or raises ValueError.
    """
    # 1. Root Schema Guard (Exact keys only)
    if list(raw.keys()) != ["files"]:
        raise ValueError("Contract root must contain exactly one key: 'files'")
    
    files = raw["files"]
    if not isinstance(files, list):
        raise ValueError("'files' must be a list")

    contract_paths = set()
    
    for i, entry in enumerate(files):
        # 2. File Entry Schema Guard (Exact keys only)
        if not isinstance(entry, dict):
            raise ValueError(f"File entry {i} must be a dictionary")
        if set(entry.keys()) != {"path", "exports"}:
            raise ValueError(f"File entry {i} must have exactly 'path' and 'exports' keys")
        
        path = entry["path"]
        exports = entry["exports"]

        if not isinstance(path, str): raise ValueError(f"Entry {i}: 'path' must be str")
        if not isinstance(exports, list): raise ValueError(f"Entry {i}: 'exports' must be list")
        
        # 3. Minimality: Must be a depended-upon file
        if path not in depended_paths:
            raise ValueError(f"File listed in contract but not depended upon in architecture: {path}")
        
        if path in contract_paths:
            raise ValueError(f"Duplicate path in contract: {path}")
        contract_paths.add(path)

        # 4. Export Integrity
        if len(exports) == 0:
            raise ValueError(f"File '{path}' listed in contract but has zero exports")

        seen_names = set()
        for j, exp in enumerate(exports):
            # 5. Export Object Schema Guard (Exact keys only)
            if not isinstance(exp, dict):
                raise ValueError(f"File '{path}': Export {j} must be a dictionary")
            if set(exp.keys()) != {"name", "inputs", "returns"}:
                raise ValueError(f"File '{path}': Export {j} must have exactly 'name', 'inputs', 'returns'")

            name = exp["name"]
            inputs = exp["inputs"]
            returns = exp["returns"]

            if not isinstance(name, str): raise ValueError(f"Export '{path}:{j}': 'name' must be str")
            if not isinstance(inputs, list): raise ValueError(f"Export '{path}:{j}': 'inputs' must be list")
            if not isinstance(returns, str): raise ValueError(f"Export '{path}:{j}': 'returns' must be str")

            # 6. Regex Hardened Signatures
            if not NAME_REGEX.match(name):
                raise ValueError(f"Invalid function name format: '{name}' in {path}")
            
            for k, inp in enumerate(inputs):
                if not isinstance(inp, str):
                    raise ValueError(f"Input {k} in '{path}:{name}' must be str")
                if not INPUT_REGEX.match(inp):
                    raise ValueError(f"Invalid input signature format: '{inp}' in {path}:{name}")
            
            # Returns regex (no whitespace)
            clean_returns = returns.strip()
            if clean_returns != returns or not RETURN_REGEX.match(returns):
                raise ValueError(f"Invalid return type format: '{returns}' in {path}:{name}")

            # 7. Uniqueness within file
            if name in seen_names:
                raise ValueError(f"Duplicate export name '{name}' in file '{path}'")
            seen_names.add(name)

    # 8. Exact Match Enforcement (Depended set == Contract set)
    missing_files = depended_paths - contract_paths
    if missing_files:
        raise ValueError(f"Missing interface contracts for depended-upon files: {missing_files}")

    return raw

# ---------------------------------------------------------------------------
# LLM Interaction
# ---------------------------------------------------------------------------

async def _run_backboard(context, user_prompt: str, model: str, depended_paths: set):
    import asyncio
    from backboard import BackboardClient
    logger = context.logger
    
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 3: BACKBOARD_API_KEY not set")
        return False

    last_error = None
    async with BackboardClient(api_key=api_key, timeout=60) as client:
        assistant = await client.create_assistant(
            name="SpecForge_Stage3",
            system_prompt=SYSTEM_PROMPT,
        )
        assistant_id = assistant.assistant_id
        logger.info(f"Stage 3: Backboard assistant created: {assistant_id}")

        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 3: Attempt {attempt}/{MAX_ATTEMPTS}")
                
                thread = await client.create_thread(assistant_id=assistant_id)
                try:
                    response = await client.add_message(
                        thread_id=thread.thread_id,
                        content=user_prompt,
                        llm_provider="openai",
                        model_name=model,
                    )
                    
                    # Hard cleaning logic (shared via Stage 1/2 pattern)
                    cleaned = response.content.strip()
                    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
                    cleaned = re.sub(r"\n```$", "", cleaned)
                    cleaned = cleaned.strip()
                    
                    parsed = json.loads(cleaned)
                    if not isinstance(parsed, dict):
                        raise ValueError("LLM output must be a JSON object (dict)")

                    validated_contracts = validate_contracts(parsed, depended_paths)
                    context.set("contracts", validated_contracts)
                    logger.info("Stage 3: Interface contracts validated and committed to state")
                    return True

                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    logger.warning(f"Stage 3: Attempt {attempt} — Failed: {e}")
                except Exception as e:
                    last_error = e
                    logger.error(f"Stage 3: Attempt {attempt} — Unexpected error: {e}")
                finally:
                    await client.delete_thread(thread.thread_id)
        finally:
            await client.delete_assistant(assistant_id)

    logger.error(f"Stage 3: Failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
    return False

def _run_openai_synchronous(context, user_prompt: str, model: str, depended_paths: set) -> bool:
    from openai import OpenAI
    logger = context.logger
    client = OpenAI()
    
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Stage 3: Attempt {attempt}/{MAX_ATTEMPTS}")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_response = response.choices[0].message.content
            
            cleaned = raw_response.strip()
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
            cleaned = cleaned.strip()
            
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("LLM output must be a JSON object (dict)")

            validated_contracts = validate_contracts(parsed, depended_paths)
            context.set("contracts", validated_contracts)
            logger.info("Stage 3: Interface contracts validated and committed to state")
            return True

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"Stage 3: Attempt {attempt} — Failed: {e}")
        except Exception as e:
            last_error = e
            logger.error(f"Stage 3: Attempt {attempt} — Unexpected error: {e}")

    logger.error(f"Stage 3: Failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
    return False

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run(context) -> bool:
    logger = context.logger
    logger.info("Stage 3: Interface Contracts started")

    # Guard: architecture must exist and have "files"
    arch = context.get("architecture")
    if not isinstance(arch, dict) or "files" not in arch:
        logger.error("Stage 3: Architecture is missing or malformed in state")
        return False
    
    # Identify depended-upon files
    depended_paths = set()
    for f in arch["files"]:
        if "depends_on" in f:
            for d in f["depends_on"]:
                depended_paths.add(d)
    
    logger.info(f"Stage 3: Identified {len(depended_paths)} depended-upon files requiring contracts")

    user_prompt = f"Architecture:\n{json.dumps(arch, indent=2)}\n\nGenerate the interface contracts."
    
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini")

    if provider == "backboard":
        import asyncio
        return asyncio.run(_run_backboard(context, user_prompt, model, depended_paths))
    else:
        return _run_openai_synchronous(context, user_prompt, model, depended_paths)
