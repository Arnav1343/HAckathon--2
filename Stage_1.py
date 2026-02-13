"""
SpecForge Stage 1 — Specification Engine

Transforms the raw user idea into a strictly structured, machine-readable
project specification. Makes zero architectural, implementation, or validation
decisions.

Responsibilities:
- Read idea_raw from state exactly as provided
- Pass through LLM with temperature=0 and JSON-only response format
- Validate output against a strict predefined schema
- Retry up to MAX_ATTEMPTS on validation failure
- Write validated spec to state
- Log attempt counts, validation failures, and final result

Does NOT:
- Generate file structures or module names
- Propose function signatures or class hierarchies
- Suggest technologies unless explicitly stated in the idea
- Infer architectural patterns
- Mutate any state beyond 'spec'
"""

import os
import re
import json
import logging


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"objective", "inputs", "outputs", "functional_requirements"}
OPTIONAL_KEYS = {"non_functional_requirements", "constraints"}
ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

FIELD_TYPES = {
    "objective": str,
    "inputs": list,
    "outputs": list,
    "functional_requirements": list,
    "non_functional_requirements": list,
    "constraints": list,
}

MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a specification extraction engine. Your sole job is to convert a raw \
project idea into a structured JSON specification.

Rules — follow these exactly:
1. Respond with a single JSON object and nothing else.
2. The JSON must contain EXACTLY these fields:
   - "objective": a single string summarizing what the project does.
   - "inputs": a list of strings describing what the system receives.
   - "outputs": a list of strings describing what the system produces.
   - "functional_requirements": a list of strings — concrete behaviors the \
system must exhibit.
   - "non_functional_requirements": a list of strings — quality attributes \
(performance, security, usability). May be empty if the idea does not imply any.
   - "constraints": a list of strings — explicit limitations or boundaries. \
May be empty if the idea does not state any.
3. Do NOT include any fields beyond the six listed above.
4. Do NOT propose or mention:
   - File names, folder structures, or module names
   - Function signatures or class hierarchies
   - Specific frameworks, libraries, or technologies UNLESS the user \
explicitly named them in the idea
   - Architectural patterns (MVC, microservices, etc.)
   - Database schemas or API endpoint paths
5. Every list field must contain at least one item, except \
non_functional_requirements and constraints which may be empty lists.
6. Use clear, declarative language. Each requirement should be independently \
understandable.
7. ANTI-INFERENCE RULE: Do not invent requirements that are not stated or \
clearly implied by the idea. If a detail is not explicitly present in the \
original idea, it MUST NOT be introduced. Choose the simplest possible \
interpretation.
8. LITERALISM: Do not improve the idea, do not add robustness, and do not \
suggest optimizations.
"""


def build_prompt(idea_raw: str) -> str:
    """Build the user-message portion of the prompt."""
    return f"Project idea:\n\n{idea_raw}"


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _call_openai(user_prompt: str, model: str, logger: logging.Logger) -> str:
    """Call OpenAI API directly."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY from env

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
# Validation
# ---------------------------------------------------------------------------

def validate_spec(raw: dict) -> dict:
    """
    Validate a parsed spec dict against the schema.

    Returns the validated (and defaulted) spec on success.
    Raises ValueError on any violation.
    """
    # 1. No extraneous keys
    extra = set(raw.keys()) - ALL_KEYS
    if extra:
        raise ValueError(f"Extraneous keys not allowed: {extra}")

    # 2. Required keys present
    missing = REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # 3. Type checking
    for key, expected_type in FIELD_TYPES.items():
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Key '{key}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        # For list fields, every element must be a string
        if expected_type is list:
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise ValueError(
                        f"Key '{key}' item {i} must be str, "
                        f"got {type(item).__name__}"
                    )

    # 4. Required list fields must be non-empty (except inputs)
    for key in ("outputs", "functional_requirements"):
        if len(raw[key]) == 0:
            raise ValueError(f"Required list '{key}' must not be empty")

    # 5. Default optional keys
    validated = dict(raw)
    for key in OPTIONAL_KEYS:
        if key not in validated:
            validated[key] = []

    return validated


def _process_llm_response(raw_response: str, logger: logging.Logger) -> dict:
    """
    Common logic to clean, parse, and validate LLM output.
    Returns validated spec dict. Raises ValueError or JSONDecodeError on failure.
    """
    if raw_response is None:
        raise ValueError("LLM returned None")

    logger.info(f"Stage 1: Received LLM response ({len(raw_response)} chars)")

    # Robust JSON cleaning: strip markdown blocks and whitespace
    # Two-step regex for safe fenced-block removal
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()

    # Parse JSON
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"Stage 1: Raw response that failed to parse: {raw_response!r}")
        raise

    # Explicit type enforcement
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object (dict)")

    # Validate schema
    return validate_spec(parsed)


def call_llm(user_prompt: str, logger: logging.Logger, model: str) -> str:
    """Call OpenAI API directly."""
    from openai import OpenAI
    client = OpenAI()  # uses OPENAI_API_KEY from env
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


async def _run_backboard(context, user_prompt: str, model: str):
    """Async execution path for Backboard provider."""
    import asyncio
    from backboard import BackboardClient
    logger = context.logger
    
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 1: BACKBOARD_API_KEY not set")
        return False

    last_error = None
    async with BackboardClient(api_key=api_key, timeout=60) as client:
        # 1. Create Assistant once per Stage 1 execution
        assistant = await client.create_assistant(
            name="SpecForge_Stage1",
            system_prompt=SYSTEM_PROMPT,
        )
        assistant_id = assistant.assistant_id
        logger.info(f"Stage 1: Backboard assistant created: {assistant_id}")

        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 1: Attempt {attempt}/{MAX_ATTEMPTS}")
                
                # 2. Create Thread per attempt for clean context
                thread = await client.create_thread(assistant_id=assistant_id)
                try:
                    response = await client.add_message(
                        thread_id=thread.thread_id,
                        content=user_prompt,
                        llm_provider="openai",
                        model_name=model,
                    )
                    raw_response = response.content
                    
                    # Common processing logic
                    validated_spec = _process_llm_response(raw_response, logger)
                    
                    # Success
                    context.set("spec", validated_spec)
                    logger.info("Stage 1: Specification validated and committed to state")
                    return True

                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    logger.warning(f"Stage 1: Attempt {attempt} — Failed: {e}")
                except Exception as e:
                    last_error = e
                    logger.error(f"Stage 1: Attempt {attempt} — Unexpected error: {e}")
                finally:
                    await client.delete_thread(thread.thread_id)
                    logger.debug(f"Stage 1: Backboard thread deleted")
        finally:
            await client.delete_assistant(assistant_id)
            logger.info("Stage 1: Backboard assistant deleted")

    logger.error(f"Stage 1: Failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
    return False


def run(context) -> bool:
    """
    Stage 1 entry point. Registered with Stage 0 orchestrator.

    Reads idea_raw, produces validated spec, writes to state.
    Returns True on success, False on failure.
    """
    logger = context.logger
    logger.info("Stage 1: Specification Engine started")

    # Guard: idea_raw must exist
    idea_raw = context.get("idea_raw")
    if not idea_raw or not isinstance(idea_raw, str) or not idea_raw.strip():
        logger.error("Stage 1: idea_raw is missing or empty")
        return False

    user_prompt = build_prompt(idea_raw)
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPEC_FORGE_MODEL", 
                           os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini"))

    if provider == "backboard":
        import asyncio
        return asyncio.run(_run_backboard(context, user_prompt, model))

    # Standard OpenAI synchronous path
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Stage 1: Attempt {attempt}/{MAX_ATTEMPTS}")
        try:
            raw_response = call_llm(user_prompt, logger, model)
            validated_spec = _process_llm_response(raw_response, logger)
            
            context.set("spec", validated_spec)
            logger.info("Stage 1: Specification validated and committed to state")
            return True

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"Stage 1: Attempt {attempt} — Failed: {e}")
        except Exception as e:
            last_error = e
            logger.error(f"Stage 1: Attempt {attempt} — Unexpected error: {e}")

    logger.error(f"Stage 1: Failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
    return False
