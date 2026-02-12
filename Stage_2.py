"""
SpecForge Stage 2 — Architecture Planning

Transforms the project specification into a macro-level architecture description
listing files, their high-level responsibilities, and their dependencies.

Responsibilities:
- Read 'spec' from state and fail if missing/malformed.
- Construct a tightly constrained prompt for macro-level architecture.
- Enforce strict JSON schema: {"files": [{"path": str, "responsibility": str, "depends_on": list[str]}]}.
- Validate paths: relative, ending in .py, unique, no '..', no absolute, no hidden.
- Validate dependencies: must exist within the architecture, no self-dependencies.
- Perform deterministic cycle detection (DFS).
- Retry up to MAX_ATTEMPTS on validation failure.
- Write validated architecture to state.

Design Principles:
- Macro-structure only.
- Zero implementation leakage (no function/class names, no code, no frameworks).
- Recursive/Circular dependency prevention.
- Deterministic failure (Strict → Reject → Retry → Fail).
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
You are an architecture planning engine. Your sole job is to define the minimal \
set of files required to satisfy a project specification and describe their \
high-level relationships.

Rules for Architecture:
1. Output MUST be a strictly valid JSON object with exactly one key: "files".
2. "files" MUST be a list of objects, each containing:
   - "path": Relative file path ending in .py (e.g., "utils.py", "core/engine.py").
   - "responsibility": A concise high-level description of what the file does.
   - "depends_on": A list of paths from this architecture that this file imports or uses.
3. ABSOLUTELY FORBIDDEN:
   - Framework names (unless specified in the spec).
   - Technology stack decisions.
   - Function signatures or class names.
   - Code snippets or internal logic.
   - Directory nesting beyond what is essential.
   - Hidden files (starting with .) or absolute paths.
   - Path traversal (..).
4. MODULARITY: Ensure files have crisp, single responsibilities.
5. DEPENDENCIES: Ensure the dependency graph is directed and acyclic.
6. ANTI-CREEP: Do not introduce utility files, config files, abstraction \
layers, test files, logging files, or service/manager/controller files \
unless the user explicitly asked for them.
7. LITERALISM: If a detail is not present in the original spec, it MUST NOT \
be introduced. Choose the simplest possible architecture.
8. Your output must be ONLY the raw JSON object. Do not provide any conversational text.\
"""

# ---------------------------------------------------------------------------
# Cycle Detection
# ---------------------------------------------------------------------------

def check_for_cycles(files: list) -> None:
    """
    Perform DFS-based cycle detection on the dependency graph.
    Raises ValueError if a cycle is detected.
    """
    adj = {f["path"]: f["depends_on"] for f in files}
    visited = set()
    stack = set()

    def visit(u, path_stack):
        if u in stack:
            cycle = " -> ".join(path_stack + [u])
            raise ValueError(f"Circular dependency detected: {cycle}")
        if u in visited:
            return

        visited.add(u)
        stack.add(u)
        path_stack.append(u)

        for v in adj.get(u, []):
            visit(v, path_stack)

        path_stack.pop()
        stack.remove(u)

    for node in adj:
        if node not in visited:
            visit(node, [])

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_architecture(raw: dict) -> dict:
    """
    Strictly validate the architecture dictionary.
    Returns the validated dict or raises ValueError.
    """
    # 1. Structure
    if set(raw.keys()) != {"files"}:
        raise ValueError("Architecture must contain exactly one key: 'files'")
    
    files = raw["files"]
    if not isinstance(files, list):
        raise ValueError("'files' must be a list")
    
    if not files:
        raise ValueError("Architecture must define at least one file")

    seen_paths = set()
    
    for i, entry in enumerate(files):
        # 2. Key enforcement
        if not isinstance(entry, dict):
             raise ValueError(f"File entry {i} must be a dictionary")
        if set(entry.keys()) != {"path", "responsibility", "depends_on"}:
            raise ValueError(f"File entry {i} has invalid or missing keys")
        
        path = entry["path"]
        resp = entry["responsibility"]
        deps = entry["depends_on"]

        # 3. Type checks
        if not isinstance(path, str): raise ValueError(f"Entry {i}: 'path' must be str")
        if not isinstance(resp, str): raise ValueError(f"Entry {i}: 'responsibility' must be str")
        if not isinstance(deps, list): raise ValueError(f"Entry {i}: 'depends_on' must be list")
        if not all(isinstance(d, str) for d in deps):
            raise ValueError(f"Entry {i}: 'depends_on' must be list of strings")

        # 4. Path safety
        if not path.endswith(".py"):
            raise ValueError(f"Entry {i}: 'path' must end in .py: {path}")
        if ".." in path:
            raise ValueError(f"Entry {i}: Path traversal (..) forbidden: {path}")
        if os.path.isabs(path):
            raise ValueError(f"Entry {i}: Absolute paths forbidden: {path}")
        if any(part.startswith(".") for part in path.split("/")):
            raise ValueError(f"Entry {i}: Hidden files/dirs forbidden: {path}")
        
        if path in seen_paths:
            raise ValueError(f"Duplicate file path: {path}")
        seen_paths.add(path)

    # 5. Dependency Integrity
    for entry in files:
        path = entry["path"]
        deps = entry["depends_on"]
        for d in deps:
            if d == path:
                raise ValueError(f"Self-dependency forbidden: '{path}'")
            if d not in seen_paths:
                raise ValueError(f"File '{path}' depends on non-existent file: '{d}'")

    # 6. Cycle Detection
    check_for_cycles(files)

    return raw

# ---------------------------------------------------------------------------
# LLM Interaction (Shared with Stage 1 logic pattern)
# ---------------------------------------------------------------------------

def _process_llm_response(raw_response: str, logger: logging.Logger) -> dict:
    if raw_response is None:
        raise ValueError("LLM returned None")

    logger.info(f"Stage 2: Received LLM response ({len(raw_response)} chars)")

    # Robust JSON cleaning: strip markdown blocks and whitespace
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"Stage 2: Raw response that failed to parse: {raw_response!r}")
        raise

    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object (dict)")

    return validate_architecture(parsed)

async def _run_backboard(context, user_prompt: str, model: str):
    import asyncio
    from backboard import BackboardClient
    logger = context.logger
    
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 2: BACKBOARD_API_KEY not set")
        return False

    last_error = None
    async with BackboardClient(api_key=api_key, timeout=60) as client:
        assistant = await client.create_assistant(
            name="SpecForge_Stage2",
            system_prompt=SYSTEM_PROMPT,
        )
        assistant_id = assistant.assistant_id
        logger.info(f"Stage 2: Backboard assistant created: {assistant_id}")

        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 2: Attempt {attempt}/{MAX_ATTEMPTS}")
                
                thread = await client.create_thread(assistant_id=assistant_id)
                try:
                    response = await client.add_message(
                        thread_id=thread.thread_id,
                        content=user_prompt,
                        llm_provider="openai",
                        model_name=model,
                    )
                    validated_arch = _process_llm_response(response.content, logger)
                    context.set("architecture", validated_arch)
                    logger.info("Stage 2: Architecture validated and committed to state")
                    return True

                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    logger.warning(f"Stage 2: Attempt {attempt} — Failed: {e}")
                except Exception as e:
                    last_error = e
                    logger.error(f"Stage 2: Attempt {attempt} — Unexpected error: {e}")
                finally:
                    await client.delete_thread(thread.thread_id)
        finally:
            await client.delete_assistant(assistant_id)

    logger.error(f"Stage 2: Failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
    return False

def call_openai_synchronous(user_prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def build_prompt(spec: dict) -> str:
    return f"Specification:\n{json.dumps(spec, indent=2)}\n\nGenerate the macro-level architecture JSON."

def run(context) -> bool:
    logger = context.logger
    logger.info("Stage 2: Architecture Planning started")

    # Guard: spec must exist
    spec = context.get("spec")
    if not spec:
        logger.error("Stage 2: Specification is missing from state")
        return False

    user_prompt = build_prompt(spec)
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini")

    if provider == "backboard":
        import asyncio
        return asyncio.run(_run_backboard(context, user_prompt, model))

    # Standard OpenAI synchronous path
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Stage 2: Attempt {attempt}/{MAX_ATTEMPTS}")
        try:
            raw_response = call_openai_synchronous(user_prompt, model)
            validated_arch = _process_llm_response(raw_response, logger)
            context.set("architecture", validated_arch)
            logger.info("Stage 2: Architecture validated and committed to state")
            return True

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"Stage 2: Attempt {attempt} — Failed: {e}")
        except Exception as e:
            last_error = e
            logger.error(f"Stage 2: Attempt {attempt} — Unexpected error: {e}")

    logger.error(f"Stage 2: Failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
    return False
