"""
SpecForge Stage 4 — Implementation Prompt Design (Compiler-Grade Freeze)

Synthesizes a structured prompt_pack JSON containing one highly precise, 
paste-ready implementation prompt per architectural file.

Responsibilities:
- Compiler-grade determinism: frozen return schemas, strict type annotations,
  explicit imports, section ordering, float precision, tie-breaking rules,
  execution flow, cross-file schema consistency, and no-additional-keys language.
- Automated Validation: reject prompts with vague adjectives, library injection,
  meta-language, shallow content, unfrozen return shapes, bare types,
  inconsistent prohibitions, or out-of-order sections.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Set, Optional

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 10
WORD_LIMIT_MAX = 300

# Vague adjectives to forbid (expanded)
VAGUE_ADJECTIVES = [
    "readable", "nice", "nicely", "clean",
    "standard", "appropriate", "efficient", "robust", "flexible",
    "extensible", "scalable", "advanced", "clear", "concise",
    "structured", "simple", "straightforward", "maintainable",
]

# Meta-language to forbid
META_LANGUAGE = [
    "pipeline", "stage", "previous", "SpecForge",
    "context", "system", "orchestrator"
]

# Baseline prohibitions — must appear identically in every prompt
BASELINE_PROHIBITIONS = [
    "No extra functions",
    "No architecture changes",
    "No logging",
    "No validation",
    "No optimization",
    "No new modules",
    "No prohibited libraries",
    "No defensive try/except blocks",
]

# Required sections in exact order
REQUIRED_SECTIONS = [
    "FILE PURPOSE:",
    "REQUIRED FUNCTION SIGNATURES:",
    "RETURN SCHEMA:",
    "INPUT/OUTPUT REQUIREMENTS:",
    "DEPENDENCY RULES:",
    "REQUIRED IMPORTS:",
]
# EXECUTION FLOW: is optional (entry-point only), inserted before WHAT NOT TO DO
# WHAT NOT TO DO: is always last
FINAL_SECTION = "WHAT NOT TO DO:"

# Common libraries to check against
COMMON_LIBS = ["argparse", "csv", "json", "os", "sys", "flask", "pandas", "numpy", "requests", "sqlite3"]

SYSTEM_PROMPT = """\
### JSON STRUCTURE
You must output a raw JSON object with this exact schema:
{{
  "prompt_pack": [
    {{
      "path": "string",
      "implementation_prompt": "string (all sections combined)"
    }}
  ]
}}

### CORE RULES
1. NO LIBRARIES: Do not reference ANY library (os, sys, csv, etc.) unless explicitly named in the Stage 1 spec.
2. LITERAL: Include all spec constraints verbatim. Choose the simplest interpretation.
3. SELF-CONTAINED: Prompt must be usable alone. No meta-language.
4. COMPILER-GRADE: Every prompt is a frozen schema. Zero interpretive freedom.

### implementation_prompt SECTIONS
The implementation_prompt string MUST contain these section labels IN THIS EXACT ORDER:

FILE PURPOSE:
  - Exact parsing mechanics: split delimiter, component count, type casts, whitespace handling.
  - No vague language like "responsible for parsing" — state exactly HOW.

REQUIRED FUNCTION SIGNATURES:
  - Every parameter and return type must use full generic types.
  - NEVER use bare `list` or `dict`. Always specify: `list[dict[str, str | float]]`, `dict[str, float]`, etc.

RETURN SCHEMA:
  - Every function returning dict or list must enumerate:
    - Exact key names
    - Exact value types per key
    - Exact key count
    - "The dictionary must contain exactly N keys. No additional keys are allowed."
  - Every function returning str (formatted output) must include:
    - Exact text template with line order
    - Separator character and exact count (e.g., "26 dashes")
    - Trailing newline policy
    - Float precision: "All float values must be formatted using exactly 2 decimal places. Use format(value, '.2f')."
  - If any "highest", "maximum", or ranking logic exists, explicitly state the tie-breaking rule (e.g., "If tied, use first occurrence" or "If tied, use lexicographically first").

INPUT/OUTPUT REQUIREMENTS:
  - Reference the exact keys from RETURN SCHEMA of upstream files.
  - If accepting a composite type, re-enumerate expected keys and types.

DEPENDENCY RULES:
  - Must list exact file dependencies from the architecture.
  - If file has dependencies, NEVER say "does not depend on any other files."
  - Must match the architecture's depends_on exactly.

REQUIRED IMPORTS:
  - One import per line using exactly: from <module> import <function>
  - No bare `import X`. No `from X import *`. No `import X as Y`.
  - If file has no dependencies, state: "No imports required."

EXECUTION FLOW: (REQUIRED for entry-point files, OMIT for library files)
  - Numbered steps with exact function call order.
  - Exact intermediate variable names.
  - Example:
    1. Parse command-line argument to get file_path.
    2. Open file_path and read lines into variable `raw_lines`.
    3. Call parse_transaction_line(line) for each line, collect into `transactions`.
    4. Call process_transactions(transactions), store result in `summary`.
    5. Call generate_report(summary), store result in `report_text`.
    6. Print report_text to stdout.

WHAT NOT TO DO:
  - Must include ALL of these EXACTLY:
    No extra functions. No architecture changes. No logging. No validation. \
No optimization. No new modules. No prohibited libraries. No defensive try/except blocks.

### CROSS-FILE CONSISTENCY
- The return schema keys of file A MUST match the input schema keys expected by file B if B depends on A.
- Variable names for intermediate data must be consistent across EXECUTION FLOW and DEPENDENCY RULES.

### DETAIL
- Maximum 300 words. Be dense and schema-like. No filler. No soft language.
- Do NOT use these words: clear, concise, structured, simple, straightforward, readable, efficient, robust, scalable, maintainable, nice, clean, appropriate, flexible, extensible, advanced.
- Output ONLY JSON. No fences.
"""

# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------

def _find_section_positions(prompt: str) -> dict[str, int]:
    """Find the character position of each section label in the prompt."""
    positions = {}
    for section in REQUIRED_SECTIONS:
        pos = prompt.find(section)
        if pos != -1:
            positions[section] = pos
    # Check for optional EXECUTION FLOW
    ef_pos = prompt.find("EXECUTION FLOW:")
    if ef_pos != -1:
        positions["EXECUTION FLOW:"] = ef_pos
    # WHAT NOT TO DO is always last
    wntd_pos = prompt.find(FINAL_SECTION)
    if wntd_pos != -1:
        positions[FINAL_SECTION] = wntd_pos
    return positions


def _validate_section_order(prompt: str, path: str) -> None:
    """Enforce that sections appear in the exact required order."""
    positions = _find_section_positions(prompt)
    
    # Build expected order: REQUIRED_SECTIONS + optional EXECUTION FLOW + WHAT NOT TO DO
    expected_order = list(REQUIRED_SECTIONS)
    if "EXECUTION FLOW:" in positions:
        expected_order.append("EXECUTION FLOW:")
    expected_order.append(FINAL_SECTION)
    
    # Filter to only sections that exist
    present_sections = [s for s in expected_order if s in positions]
    
    # Check monotonically increasing positions
    for i in range(len(present_sections) - 1):
        curr = present_sections[i]
        nxt = present_sections[i + 1]
        if positions[curr] >= positions[nxt]:
            raise ValueError(
                f"Prompt for '{path}': Section '{nxt}' must appear after '{curr}'. "
                f"Sections are out of order."
            )


def _validate_return_schema_freeze(prompt: str, path: str) -> None:
    """Enforce that return dicts/lists have frozen schemas with key enumeration."""
    prompt_lower = prompt.lower()
    
    # Check for bare -> dict or -> list (without type parameters)
    # Match -> dict NOT followed by [ (allowing whitespace)
    bare_dict = re.search(r'->\s*dict(?!\s*\[)', prompt)
    if bare_dict:
        raise ValueError(
            f"Prompt for '{path}': Contains bare '-> dict' without type parameters. "
            f"Must use e.g. '-> dict[str, str | float]' with full key enumeration."
        )
    
    bare_list = re.search(r'->\s*list(?!\s*\[)', prompt)
    if bare_list:
        raise ValueError(
            f"Prompt for '{path}': Contains bare '-> list' without type parameters. "
            f"Must use e.g. '-> list[dict[str, str | float]]'."
        )
    
    # If return type involves dict, must have key enumeration
    # Only check within the RETURN SCHEMA section itself (not the full prompt)
    if 'return schema:' in prompt_lower:
        schema_section = prompt[prompt.find("RETURN SCHEMA:"):]
        # Truncate at next section
        for s in ["INPUT/OUTPUT REQUIREMENTS:", "DEPENDENCY RULES:", "REQUIRED IMPORTS:",
                   "EXECUTION FLOW:", FINAL_SECTION]:
            idx = schema_section.find(s)
            if idx > 0:
                schema_section = schema_section[:idx]
                break
        
        schema_lower = schema_section.lower()
        
        # Only enforce key enumeration if the RETURN SCHEMA section itself references dict
        if 'dict' in schema_lower:
            if "exactly" not in schema_lower:
                raise ValueError(
                    f"Prompt for '{path}': RETURN SCHEMA must contain 'exactly N keys' language."
                )
            if "no additional keys" not in schema_lower:
                raise ValueError(
                    f"Prompt for '{path}': RETURN SCHEMA must contain 'No additional keys' language."
                )


def _validate_float_precision(prompt: str, path: str) -> None:
    """If float appears in return schema and output involves formatting, must freeze precision."""
    prompt_lower = prompt.lower()
    
    has_float_return = 'float' in prompt_lower and 'return schema:' in prompt_lower
    has_str_return = '-> str' in prompt_lower
    
    if has_float_return and has_str_return:
        if 'decimal place' not in prompt_lower and '.2f' not in prompt and '.0f' not in prompt:
            raise ValueError(
                f"Prompt for '{path}': Contains float in return schema and str output "
                f"but does not freeze float precision. Must specify decimal places "
                f"and format specifier (e.g., format(value, '.2f'))."
            )


def _validate_tie_breaking(prompt: str, path: str) -> None:
    """If ranking/max logic exists, must define tie-breaking."""
    prompt_lower = prompt.lower()
    ranking_keywords = ["highest", "maximum", "largest", "greatest"]
    
    # Use word boundaries to avoid false positives (e.g., "top_user" matching "top")
    has_ranking = any(re.search(rf'\b{kw}\b', prompt_lower) for kw in ranking_keywords)
    if has_ranking and "tie" not in prompt_lower:
        raise ValueError(
            f"Prompt for '{path}': Contains ranking logic (highest/maximum) "
            f"but does not define tie-breaking rule. Must explicitly state tie resolution."
        )


def _validate_import_strictness(prompt: str, path: str) -> None:
    """Enforce import format: only 'from X import Y'. No bare imports, wildcards, or aliases."""
    # Find REQUIRED IMPORTS section
    imports_start = prompt.find("REQUIRED IMPORTS:")
    if imports_start == -1:
        return  # Section missing is caught elsewhere
    
    # Extract section content (after the label)
    imports_section = prompt[imports_start + len("REQUIRED IMPORTS:"):]
    for s in ["EXECUTION FLOW:", FINAL_SECTION]:
        idx = imports_section.find(s)
        if idx > 0:
            imports_section = imports_section[:idx]
            break
    
    imports_lower = imports_section.lower().strip()
    
    # Skip if no imports required
    if "no imports required" in imports_lower:
        return
    
    # Check for bare import: "import X" NOT preceded by "from"
    # Match word boundary "import" not preceded by "from "
    if re.search(r'(?<!\bfrom\s)(?<!\w)import\s+\w+', imports_section):
        # Double check it's not a "from X import Y" by looking more carefully
        # Find all "import" occurrences and check if preceded by "from"
        for m in re.finditer(r'\bimport\s+(\w+)', imports_section):
            # Check if "from" appears before this "import" in the same statement
            before = imports_section[:m.start()].rstrip()
            # Look for "from X" pattern immediately before
            if not re.search(r'\bfrom\s+\w+\s*$', before):
                raise ValueError(
                    f"Prompt for '{path}': REQUIRED IMPORTS contains bare 'import {m.group(1)}'. "
                    f"Must use 'from X import Y'."
                )
    
    # Check for wildcard imports
    if re.search(r'\bimport\s+\*', imports_section):
        raise ValueError(
            f"Prompt for '{path}': REQUIRED IMPORTS contains wildcard import. "
            f"Must use explicit 'from X import Y'."
        )
    
    # Check for alias imports
    if re.search(r'\bimport\s+\w+\s+as\s+\w+', imports_section):
        raise ValueError(
            f"Prompt for '{path}': REQUIRED IMPORTS contains alias import. "
            f"No 'as' aliases allowed."
        )


def _validate_execution_flow(prompt: str, path: str, arch: dict) -> None:
    """Entry-point files (no dependents) must have EXECUTION FLOW section."""
    # Determine if this file is an entry point (no other file depends on it)
    dependents = set()
    for f in arch.get("files", []):
        for dep in f.get("depends_on", []):
            dependents.add(dep)
    
    is_entry_point = path not in dependents
    
    if is_entry_point and "EXECUTION FLOW:" not in prompt:
        raise ValueError(
            f"Prompt for '{path}': Entry-point file must contain 'EXECUTION FLOW:' section "
            f"with numbered steps defining exact call order and variable names."
        )


def _validate_dependency_crosscheck(prompt: str, path: str, arch: dict) -> None:
    """Cross-reference DEPENDENCY RULES against architecture depends_on."""
    # Find this file's dependencies in architecture
    file_deps = []
    for f in arch.get("files", []):
        if f["path"] == path:
            file_deps = f.get("depends_on", [])
            break
    
    prompt_lower = prompt.lower()
    dep_section_start = prompt.find("DEPENDENCY RULES:")
    if dep_section_start == -1:
        return
    
    dep_section = prompt[dep_section_start:]
    for s in ["REQUIRED IMPORTS:", "EXECUTION FLOW:", FINAL_SECTION]:
        idx = dep_section.find(s)
        if idx > 0:
            dep_section = dep_section[:idx]
            break
    
    dep_lower = dep_section.lower()
    
    if file_deps:
        # File HAS dependencies — must NOT say "does not depend"
        if "does not depend" in dep_lower:
            raise ValueError(
                f"Prompt for '{path}': DEPENDENCY RULES says 'does not depend' "
                f"but architecture declares dependencies: {file_deps}"
            )
        # Each dependency should be mentioned
        for dep in file_deps:
            dep_name = dep.replace(".py", "")
            if dep_name not in dep_lower and dep not in dep_lower:
                raise ValueError(
                    f"Prompt for '{path}': DEPENDENCY RULES does not mention "
                    f"declared dependency '{dep}'"
                )
    else:
        # File has NO dependencies — should not reference other files as deps
        pass


def _validate_cross_file_schema(pack: list, path: str, arch: dict) -> None:
    """Validate that return schema keys of dependency files are referenced in dependent file's INPUT/OUTPUT section."""
    # Find this file's dependencies
    file_deps = []
    for f in arch.get("files", []):
        if f["path"] == path:
            file_deps = f.get("depends_on", [])
            break
    
    if not file_deps:
        return
    
    # Determine if this file is an entry-point (no other file depends on it)
    # Entry-points orchestrate — they chain calls but don't directly consume internal keys
    dependents = set()
    for f in arch.get("files", []):
        for dep in f.get("depends_on", []):
            dependents.add(dep)
    
    is_entry_point = path not in dependents
    if is_entry_point:
        return  # Entry-points don't need to re-enumerate intermediate schema keys
    
    # Get current file's INPUT/OUTPUT REQUIREMENTS section
    curr_prompt = None
    for entry in pack:
        if entry["path"] == path:
            curr_prompt = entry["implementation_prompt"]
            break
    
    if not curr_prompt:
        return
    
    io_start = curr_prompt.find("INPUT/OUTPUT REQUIREMENTS:")
    if io_start == -1:
        return
    
    io_section = curr_prompt[io_start:]
    for s in ["DEPENDENCY RULES:", "REQUIRED IMPORTS:", "EXECUTION FLOW:", FINAL_SECTION]:
        idx = io_section.find(s)
        if idx > 0:
            io_section = io_section[:idx]
            break
    
    io_lower = io_section.lower()
    
    # For each dependency, extract declared keys from its RETURN SCHEMA
    for dep_path in file_deps:
        dep_prompt = None
        for entry in pack:
            if entry["path"] == dep_path:
                dep_prompt = entry["implementation_prompt"]
                break
        
        if not dep_prompt:
            continue
        
        # Extract keys from dependency's RETURN SCHEMA
        dep_schema_start = dep_prompt.find("RETURN SCHEMA:")
        if dep_schema_start == -1:
            continue
        
        dep_schema = dep_prompt[dep_schema_start:]
        for s in ["INPUT/OUTPUT REQUIREMENTS:", "DEPENDENCY RULES:", "REQUIRED IMPORTS:",
                   "EXECUTION FLOW:", FINAL_SECTION]:
            idx = dep_schema.find(s)
            if idx > 0:
                dep_schema = dep_schema[:idx]
                break
        
        # Extract quoted key names: "key_name"
        dep_keys = re.findall(r'"(\w+)"', dep_schema)
        
        if not dep_keys:
            continue
        
        missing_keys = [k for k in dep_keys if k.lower() not in io_lower]
        if missing_keys:
            raise ValueError(
                f"Prompt for '{path}': Cross-file schema mismatch. "
                f"Dependency '{dep_path}' declares keys {dep_keys} but "
                f"'{path}' INPUT/OUTPUT does not reference: {missing_keys}"
            )


def validate_prompt_pack(raw: dict, arch_paths: set, spec: dict, arch: dict = None) -> dict:
    """
    Compiler-grade validation for prompt_pack.
    Checks: section order, return schema freeze, float precision, tie-breaking,
    import strictness, dependency cross-check, execution flow, cross-file schema,
    word limits, prohibited libraries, meta-language, vagueness, and prohibition parity.
    """
    if set(raw.keys()) != {"prompt_pack"}:
        raise ValueError(f"Root must contain exactly one key: 'prompt_pack'. Found: {list(raw.keys())}")
    
    pack = raw["prompt_pack"]
    if not isinstance(pack, list):
        raise ValueError("'prompt_pack' must be a list")

    # Extract allowed libraries from spec
    spec_text = json.dumps(spec).lower()
    allowed_libs = [lib for lib in COMMON_LIBS if lib in spec_text]

    pack_paths = set()

    for i, entry in enumerate(pack):
        if not isinstance(entry, dict) or set(entry.keys()) != {"path", "implementation_prompt"}:
            keys_found = list(entry.keys()) if isinstance(entry, dict) else "Not a dict"
            raise ValueError(f"Entry {i} must have exactly 'path' and 'implementation_prompt'. Found: {keys_found}")
        
        path = entry["path"]
        prompt = entry["implementation_prompt"]

        if path not in arch_paths:
            raise ValueError(f"Prompt provided for path not in architecture: {path}")
        
        if path in pack_paths:
            raise ValueError(f"Duplicate prompt for path: {path}")
        pack_paths.add(path)

        # --- 1. Section Presence Check ---
        all_required = REQUIRED_SECTIONS + [FINAL_SECTION]
        for section in all_required:
            if section not in prompt:
                snippet = prompt[:100].replace('\n', ' ')
                raise ValueError(f"Prompt for '{path}' is missing mandatory section: {section}. Snippet: [{snippet}...]")

        # --- 2. Section Order Check ---
        _validate_section_order(prompt, path)

        # --- 3. Word Count Check (no minimum, max only) ---
        word_count = len(prompt.split())
        if word_count > WORD_LIMIT_MAX:
            raise ValueError(f"Prompt for '{path}' exceeds {WORD_LIMIT_MAX} words (Count: {word_count})")

        # --- 4. Return Schema Freeze ---
        _validate_return_schema_freeze(prompt, path)

        # --- 5. Float Precision ---
        _validate_float_precision(prompt, path)

        # --- 6. Tie-Breaking ---
        _validate_tie_breaking(prompt, path)

        # --- 7. Import Strictness ---
        _validate_import_strictness(prompt, path)

        # --- 8. Library Injection Check ---
        prompt_lower = prompt.lower()
        for lib in COMMON_LIBS:
            if lib not in allowed_libs:
                pattern = re.compile(rf'\b{lib}\b', re.IGNORECASE)
                if pattern.search(prompt_lower):
                    raise ValueError(f"Prompt for '{path}' injects unrequested library: {lib}")

        # --- 9. Meta-language Check ---
        for meta in META_LANGUAGE:
            if meta.lower() in prompt_lower:
                raise ValueError(f"Prompt for '{path}' contains prohibited meta-language: '{meta}'")

        # --- 10. Vague Adjective Check ---
        for adj in VAGUE_ADJECTIVES:
            pattern = re.compile(rf'\b{adj}\b', re.IGNORECASE)
            if pattern.search(prompt_lower):
                raise ValueError(f"Prompt for '{path}' contains vague adjective: '{adj}'")

        # --- 11. Baseline Prohibitions Parity ---
        for b in BASELINE_PROHIBITIONS:
            if b.lower() not in prompt_lower:
                raise ValueError(f"Prompt for '{path}' missing baseline prohibition: '{b}'")

        # --- 12. Execution Flow (entry-point check) ---
        if arch:
            _validate_execution_flow(prompt, path, arch)

        # --- 13. Dependency Cross-Check ---
        if arch:
            _validate_dependency_crosscheck(prompt, path, arch)

    # --- 14. Cross-File Schema Consistency ---
    if arch:
        for entry in pack:
            _validate_cross_file_schema(pack, entry["path"], arch)

    # Parity check: all architecture paths must have prompts
    missing = arch_paths - pack_paths
    if missing:
        raise ValueError(f"Missing implementation prompts for files: {missing}")

    return raw


def extract_json(text: str, logger=None) -> dict:
    """Robustly extract JSON from potentially messy LLM response."""
    # Surgical extract: everything between first { and last } (inclusive)
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"Surgical JSON extraction failed: {e}. Content: [{json_str[:100]}...]")
    
    # Surgical extract: everything between first [ and last ] (inclusive)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"Surgical List extraction failed: {e}. Content: [{json_str[:100]}...]")
                
    # Fallback to strip
    return json.loads(text.strip())

# ---------------------------------------------------------------------------
# Provider Interactions
# ---------------------------------------------------------------------------

async def _run_backboard(context, user_prompt: str, model: str, arch_paths: set, spec: dict, arch: dict):
    from backboard import BackboardClient
    logger = context.logger
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 4: BACKBOARD_API_KEY not set")
        return False

    async with BackboardClient(api_key=api_key, timeout=60) as client:
        assistant = await client.create_assistant(
            name="SpecForge_Stage4_Hardened",
            system_prompt=SYSTEM_PROMPT,
        )
        try:
            thread = await client.create_thread(assistant_id=assistant.assistant_id)
            current_user_prompt = user_prompt
            
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 4: Attempt {attempt}/{MAX_ATTEMPTS}")
                try:
                    response = await client.add_message(
                        thread_id=thread.thread_id,
                        content=current_user_prompt,
                        llm_provider="openai",
                        model_name=model,
                    )
                    content = response.content.strip()
                    logger.info(f"Stage 4 Debug: Raw content length: {len(content)}")
                    logger.info(f"Stage 4 Debug: FULL RAW CONTENT: [{content}]")
                    
                    if not content:
                        raise ValueError("LLM returned empty response")
                    
                    parsed = extract_json(content, logger)
                    validated = validate_prompt_pack(parsed, arch_paths, spec, arch)
                    context.set("prompt_pack", validated)
                    logger.info("Stage 4: Compiler-grade implementation prompt pack validated")
                    return True
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Stage 4: Attempt {attempt} failed: {e}")
                    if isinstance(e, json.JSONDecodeError):
                        logger.debug(f"Raw content: {content[:500]}...")
                    
                    current_user_prompt = f"The previous attempt failed validation with the following error:\n{e}\n\nPlease regenerate the JSON correctly. FIX the error and follow ALL rules."
        finally:
            await client.delete_assistant(assistant.assistant_id)
    return False

def _run_openai_synchronous(context, user_prompt: str, model: str, arch_paths: set, spec: dict, arch: dict) -> bool:
    from openai import OpenAI
    logger = context.logger
    client = OpenAI()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Stage 4: Attempt {attempt}/{MAX_ATTEMPTS}")
        content = ""
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response")
                
            parsed = extract_json(content)
            validated = validate_prompt_pack(parsed, arch_paths, spec, arch)
            context.set("prompt_pack", validated)
            logger.info("Stage 4: Compiler-grade implementation prompt pack validated")
            return True
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Stage 4: Attempt {attempt} failed: {e}")
            if isinstance(e, json.JSONDecodeError):
                logger.debug(f"Raw content: {content[:500]}...")
                
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Validation failed: {e}. Please fix and regenerate."})
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
    
    user_prompt = f"Specification:\n{json.dumps(spec, indent=2)}\n\nArchitecture:\n{json.dumps(arch, indent=2)}\n\nStructural Decomposition:\n{json.dumps(contracts, indent=2)}\n\nSynthesize the compiler-grade implementation prompt pack."
    
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini")

    if provider == "backboard":
        import asyncio
        return asyncio.run(_run_backboard(context, user_prompt, model, arch_paths, spec, arch))
    else:
        return _run_openai_synchronous(context, user_prompt, model, arch_paths, spec, arch)
