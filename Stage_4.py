"""
SpecForge Stage 4 — Implementation Prompt Design

Synthesizes a structured prompt_pack JSON containing one minimal, 
copy-paste-ready implementation prompt per architectural file.

Responsibilities:
- Generate one prompt per file in the architecture.
- Enforce a standardized textual template.
- Enforce Anti-Expansion: No new architectural elements, no technologies, 
  no performance sugar, no unrequested error handling.
- Enforce Literalism: Simplest possible interpretation.
- Enforce Conciseness: < 300 words per prompt.
- Model-Agnostic: No references to LLM behavior.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3
WORD_LIMIT = 300

SYSTEM_PROMPT = """\
You are an implementation prompt synthesis engine. Your sole job is to \
generate a structured prompt_pack JSON containing implementation prompts \
for a development team.

You must NOT generate executable code. You must NOT suggest improvements.

Rules for Prompt Design:
1. Output MUST be a strictly valid JSON object with exactly one key: "prompt_pack".
2. "prompt_pack" MUST be a list of objects, each containing:
   - "path": The file path from the architecture.
   - "implementation_prompt": The synthesized instruction text.
3. Every prompt MUST follow this exact template:
   FILE PURPOSE:
   REQUIRED FUNCTION SIGNATURES: (Textual only)
   INPUT/OUTPUT REQUIREMENTS:
   DEPENDENCY RULES:
   WHAT NOT TO DO:
4. ANTI-EXPANSION:
   - DO NOT introduce new architectural elements (utility files, managers).
   - DO NOT suggest frameworks, technologies, or libraries unless explicitly named.
   - DO NOT add performance sugar (caching, async, threads, optimizations).
   - DO NOT add defensive sugar (try/except, validation) unless "robustness" was requested.
5. LITERALISM: Simplest possible interpretation. No assumption filling.
6. CONCISENESS: Each prompt must be under 300 words. Imperative tone.
7. MODEL-AGNOSTIC: No references to LLM behavior, tokens, or parameters.
8. Your output must be ONLY the raw JSON object. Do not provide any conversational text.\
"""

# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------

def validate_prompt_pack(raw: dict, arch_paths: set) -> dict:
    """
    Validate the prompt_pack structure, completeness, and constraints.
    """
    if set(raw.keys()) != {"prompt_pack"}:
        raise ValueError("Root must contain exactly one key: 'prompt_pack'")
    
    pack = raw["prompt_pack"]
    if not isinstance(pack, list):
        raise ValueError("'prompt_pack' must be a list")

    pack_paths = set()
    required_sections = [
        "FILE PURPOSE:",
        "REQUIRED FUNCTION SIGNATURES:",
        "INPUT/OUTPUT REQUIREMENTS:",
        "DEPENDENCY RULES:",
        "WHAT NOT TO DO:"
    ]

    for i, entry in enumerate(pack):
        if not isinstance(entry, dict) or set(entry.keys()) != {"path", "implementation_prompt"}:
            raise ValueError(f"Entry {i} must have exactly 'path' and 'implementation_prompt'")
        
        path = entry["path"]
        prompt = entry["implementation_prompt"]

        if path not in arch_paths:
            raise ValueError(f"Prompt provided for path not in architecture: {path}")
        
        if path in pack_paths:
            raise ValueError(f"Duplicate prompt for path: {path}")
        pack_paths.add(path)

        # Content validation
        for section in required_sections:
            if section not in prompt:
                raise ValueError(f"Prompt for '{path}' is missing mandatory section: {section}")
        
        # Word count check
        word_count = len(prompt.split())
        if word_count > WORD_LIMIT:
            raise ValueError(f"Prompt for '{path}' exceeds {WORD_LIMIT} words (Count: {word_count})")

    # Parity check
    missing = arch_paths - pack_paths
    if missing:
        raise ValueError(f"Missing implementation prompts for files: {missing}")

    return raw

# ---------------------------------------------------------------------------
# Provider Interactions
# ---------------------------------------------------------------------------

async def _run_backboard(context, user_prompt: str, model: str, arch_paths: set):
    from backboard import BackboardClient
    logger = context.logger
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 4: BACKBOARD_API_KEY not set")
        return False

    async with BackboardClient(api_key=api_key, timeout=60) as client:
        assistant = await client.create_assistant(
            name="SpecForge_Stage4",
            system_prompt=SYSTEM_PROMPT,
        )
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 4: Attempt {attempt}/{MAX_ATTEMPTS}")
                thread = await client.create_thread(assistant_id=assistant.assistant_id)
                try:
                    response = await client.add_message(
                        thread_id=thread.thread_id,
                        content=user_prompt,
                        llm_provider="openai",
                        model_name=model,
                    )
                    cleaned = response.content.strip()
                    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
                    cleaned = re.sub(r"\n```$", "", cleaned)
                    
                    parsed = json.loads(cleaned)
                    validated = validate_prompt_pack(parsed, arch_paths)
                    context.set("prompt_pack", validated)
                    logger.info("Stage 4: Implementation prompt pack validated")
                    return True
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Stage 4: Attempt {attempt} failed: {e}")
                finally:
                    await client.delete_thread(thread.thread_id)
        finally:
            await client.delete_assistant(assistant.assistant_id)
    return False

def _run_openai_synchronous(context, user_prompt: str, model: str, arch_paths: set) -> bool:
    from openai import OpenAI
    logger = context.logger
    client = OpenAI()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Stage 4: Attempt {attempt}/{MAX_ATTEMPTS}")
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
            parsed = json.loads(response.choices[0].message.content)
            validated = validate_prompt_pack(parsed, arch_paths)
            context.set("prompt_pack", validated)
            logger.info("Stage 4: Implementation prompt pack validated")
            return True
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Stage 4: Attempt {attempt} failed: {e}")
    return False

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run(context) -> bool:
    logger = context.logger
    logger.info("Stage 4: Implementation Prompt Design started")

    spec = context.get("spec")
    arch = context.get("architecture")
    contracts = context.get("contracts")

    if not all([spec, arch, contracts]):
        logger.error("Stage 4: Missing required context (spec, arch, or contracts)")
        return False
    
    arch_paths = {f["path"] for f in arch["files"]}
    
    user_prompt = f"Specification:\n{json.dumps(spec, indent=2)}\n\nArchitecture:\n{json.dumps(arch, indent=2)}\n\nStructural Decomposition:\n{json.dumps(contracts, indent=2)}\n\nGenerate the implementation prompt pack."
    
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini")

    if provider == "backboard":
        import asyncio
        return asyncio.run(_run_backboard(context, user_prompt, model, arch_paths))
    else:
        return _run_openai_synchronous(context, user_prompt, model, arch_paths)
