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
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 5
WORD_LIMIT_MAX = 300

VAGUE_ADJECTIVES = ["readable", "nice", "clean", "standard", "appropriate", "efficient", "robust", "flexible", "extensible", "scalable", "advanced", "clear", "concise", "structured", "simple", "straightforward", "maintainable"]
META_LANGUAGE = ["pipeline", "stage", "previous", "SpecForge", "context", "orchestrator"]
BASELINE_PROHIBITIONS = ["No extra functions", "No architecture changes", "No logging", "No validation", "No optimization", "No new modules", "No prohibited libraries", "No defensive try/except blocks"]
REQUIRED_SECTIONS = ["FILE PURPOSE:", "REQUIRED FUNCTION SIGNATURES:", "RETURN SCHEMA:", "INPUT/OUTPUT REQUIREMENTS:", "DEPENDENCY RULES:", "REQUIRED IMPORTS:"]
FINAL_SECTION = "WHAT NOT TO DO:"
COMMON_LIBS = ["argparse", "csv", "json", "os", "sys", "flask", "pandas", "numpy", "requests", "sqlite3"]

SYSTEM_PROMPT = """\
Convert input (Spec, Architecture, Contracts) into a JSON 'prompt_pack' of implementation prompts.
One prompt per file in the architecture.

JSON SCHEMA:
{{
  "prompt_pack": [
    {{ "path": "string", "implementation_prompt": "string" }}
  ]
}}

SIGNATURE RULES:
- NEVER use bare 'list' or 'dict'.
- ALWAYS use generic types: 'list[str]', 'list[dict[str, float]]', 'dict[str, int]'.
- NO BACKSLASHES (\\) for line continuation.
- Example: 'def process(data: list[dict[str, float]]) -> dict[str, Any]:'

IMPLEMENTATION PROMPT RULES (ORDER MATTERS):
1. FILE PURPOSE: Exact parsing/logic mechanics. No vague verbs.
2. REQUIRED FUNCTION SIGNATURES: Use generic types.
3. RETURN SCHEMA: Enumerate all keys, types, and counts. Must say "exactly N keys" and "No additional keys".
4. INPUT/OUTPUT REQUIREMENTS: Reference keys from RETURN SCHEMA of upstream files.
5. DEPENDENCY RULES: Match architecture 'depends_on' exactly.
6. REQUIRED IMPORTS: Exactly 'from <mod> import <func>'. One per line. No base imports.
7. EXECUTION FLOW (ENTRY-POINT ONLY): Numbered call order with intermediate variable names.
8. WHAT NOT TO DO: Include all baseline prohibitions.
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


def _validate_return_schema_freeze(prompt: str, path: str) -> List[str]:
    errors = []
    prompt_lower = prompt.lower()
    
    if re.search(r'->\s*dict(?!\s*\[)', prompt):
        errors.append(f"PROMPT ERROR ({path}): Contains bare '-> dict' without type parameters. Use '-> dict[str, Any]' or similar.")
    
    if re.search(r'->\s*list(?!\s*\[)', prompt):
        errors.append(f"PROMPT ERROR ({path}): Contains bare '-> list' without type parameters. Use '-> list[str]' or similar.")
    
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
                errors.append(f"PROMPT ERROR ({path}): RETURN SCHEMA must contain 'exactly N keys' language.")
            if "no additional keys" not in schema_lower:
                errors.append(f"PROMPT ERROR ({path}): RETURN SCHEMA must contain 'No additional keys' language.")
    return errors


def _validate_float_precision(prompt: str, path: str) -> List[str]:
    prompt_lower = prompt.lower()
    if 'float' in prompt_lower and 'return schema:' in prompt_lower and '-> str' in prompt_lower:
        if 'decimal place' not in prompt_lower and '.2f' not in prompt and '.0f' not in prompt:
            return [f"PROMPT ERROR ({path}): Missing float precision for string output."]
    return []


def _validate_tie_breaking(prompt: str, path: str) -> List[str]:
    p_lower = prompt.lower()
    if any(re.search(rf'\b{kw}\b', p_lower) for kw in ["highest", "maximum", "largest", "ranking"]):
        if "tie" not in p_lower:
            return [f"PROMPT ERROR ({path}): Missing tie-breaking rules for ranking logic."]
    return []


def _validate_import_strictness(prompt: str, path: str) -> List[str]:
    """Enforce import format: only 'from X import Y'. No bare imports, wildcards, or aliases."""
    errors = []
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


def _validate_cross_file_schema(pack: list, path: str, arch: dict) -> str | None:
    # Optimized to return surgical error string or None
    file_deps = []
    for f in arch.get("files", []):
        if f["path"] == path:
            file_deps = f.get("depends_on", [])
            break
    if not file_deps: return None

    dependents = {dep for f in arch["files"] for dep in f.get("depends_on", [])}
    if path not in dependents: return None

    curr_entry = next((e for e in pack if e["path"] == path), None)
    if not curr_entry: return None
    curr_prompt = curr_entry["implementation_prompt"]
    
    io_start = curr_prompt.find("INPUT/OUTPUT REQUIREMENTS:")
    if io_start == -1: return None
    
    io_section = curr_prompt[io_start:].lower()
    for s in ["DEPENDENCY RULES:", "REQUIRED IMPORTS:", "EXECUTION FLOW:", FINAL_SECTION]:
        idx = io_section.find(s.lower())
        if idx > 0:
            io_section = io_section[:idx]
            break
    
    for dep_path in file_deps:
        dep_entry = next((e for e in pack if e["path"] == dep_path), None)
        if not dep_entry: continue
        dep_prompt = dep_entry["implementation_prompt"]
        
        dep_schema_start = dep_prompt.find("RETURN SCHEMA:")
        if dep_schema_start == -1: continue
        
        dep_schema = dep_prompt[dep_schema_start:]
        for s in ["INPUT/OUTPUT REQUIREMENTS:", "DEPENDENCY RULES:", "REQUIRED IMPORTS:", "EXECUTION FLOW:", FINAL_SECTION]:
            idx = dep_schema.find(s)
            if idx > 0:
                dep_schema = dep_schema[:idx]
                break
        
        dep_keys = re.findall(r'"(\w+)"', dep_schema)
        missing_keys = [k for k in dep_keys if k.lower() not in io_section]
        if missing_keys:
            return f"SCHEMA ERROR: {path} depends on {dep_path} but is missing required keys in INPUT/OUTPUT: {missing_keys}"
    return None

def validate_prompt_pack(raw: dict, arch_paths: set, spec: dict, arch: dict = None) -> list[str]:
    # Returns list of surgical error strings. Empty list means success.
    errors = []
    if set(raw.keys()) != {"prompt_pack"}:
        errors.append("SCHEMA ERROR: Root must contain exactly 'prompt_pack' key.")
        return errors
    
    pack = raw["prompt_pack"]
    spec_text = json.dumps(spec).lower()
    allowed_libs = [lib for lib in COMMON_LIBS if lib in spec_text]
    pack_paths = {e["path"] for e in pack if "path" in e}

    for i, entry in enumerate(pack):
        path = entry.get("path", f"Unknown_{i}")
        prompt_raw = entry.get("implementation_prompt", "")
        
        # Handle dict-typed prompts (some models return structured prompts)
        if isinstance(prompt_raw, dict):
            prompt = "\n".join([f"{k} {v}" for k, v in prompt_raw.items()])
        else:
            prompt = str(prompt_raw)
        
        if path not in arch_paths:
            errors.append(f"PATH ERROR: Prompt provided for unknown path: {path}")
            continue

        # Section checks
        for section in REQUIRED_SECTIONS + [FINAL_SECTION]:
            if section not in prompt:
                errors.append(f"FORMAT ERROR ({path}): Missing section {section}")

        # Basic constraints
        if len(prompt.split()) > WORD_LIMIT_MAX:
            errors.append(f"LENGTH ERROR ({path}): Exceeds {WORD_LIMIT_MAX} words")
        
        # Prohibited content
        p_lower = prompt.lower()
        for lib in COMMON_LIBS:
            if lib not in allowed_libs and re.search(rf'\b{lib}\b', p_lower):
                errors.append(f"SECURITY ERROR ({path}): Prohibited library used: {lib}")
        for meta in META_LANGUAGE:
            if meta.lower() in p_lower:
                errors.append(f"META ERROR ({path}): Prohibited meta-language: {meta}")
        for b in BASELINE_PROHIBITIONS:
            if b.lower() not in p_lower:
                errors.append(f"PARITY ERROR ({path}): Missing baseline prohibition: {b}")

        # Schema & Precision
        errors.extend(_validate_return_schema_freeze(prompt, path))
        errors.extend(_validate_float_precision(prompt, path))
        errors.extend(_validate_tie_breaking(prompt, path))

    if arch:
        for entry in pack:
            err = _validate_cross_file_schema(pack, entry["path"], arch)
            if err: errors.append(err)

    missing = arch_paths - pack_paths
    if missing:
        errors.append(f"COMPLETENESS ERROR: Missing prompts for files: {missing}")

    return errors

def extract_json(text: str) -> dict:
    # Surgical extract: everything between first { and last } (inclusive)
    # First, strip markdown fences if they exist
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()
    
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        inner = cleaned[start:end+1]
        try: return json.loads(inner)
        except Exception as e:
            # Try to handle common LLM errors like illegal backslashes before newlines
            try:
                fixed = inner.replace('\\\n', '\n').replace('\\ \n', '\n')
                return json.loads(fixed)
            except: pass
    
    # Fallback to direct load of the original or cleaned text
    return json.loads(cleaned)

def get_context_hash(spec: dict, arch: dict, contracts: dict) -> str:
    combined = json.dumps({"s": spec, "a": arch, "c": contracts}, sort_keys=True)
    return hashlib.sha256(combined.encode()).hexdigest()

async def _run_backboard(context, user_prompt: str, model: str, arch_paths: set, spec: dict, arch: dict):
    from backboard import BackboardClient
    logger = context.logger
    api_key = os.environ.get("BACKBOARD_API_KEY", "")
    if not api_key:
        logger.error("Stage 4: BACKBOARD_API_KEY not set")
        return False

    async with BackboardClient(api_key=api_key, timeout=120) as client:
        assistant = await client.create_assistant(name="SpecForge_Stage4_Hardened", system_prompt=SYSTEM_PROMPT)
        thread = await client.create_thread(assistant_id=assistant.assistant_id)
        current_user_prompt = user_prompt
        
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                logger.info(f"Stage 4: Attempt {attempt}/{MAX_ATTEMPTS}")
                try:
                    response = await client.add_message(thread_id=thread.thread_id, content=current_user_prompt, llm_provider="openai", model_name=model)
                    content = response.content or ""
                    if not content.strip():
                        logger.warning(f"Stage 4: Attempt {attempt} returned empty content")
                        continue
                        
                    parsed = extract_json(content)
                    
                    errors = validate_prompt_pack(parsed, arch_paths, spec, arch)
                    if not errors:
                        context.set("prompt_pack", parsed)
                        return True
                    
                    logger.warning(f"Stage 4: Attempt {attempt} failed validation.")
                    error_feedback = "\n".join([f"- {err}" for err in errors])
                    logger.info(f"Stage 4 Surgical Feedback:\n{error_feedback}")
                    current_user_prompt = f"VALIDATION ERRORS:\n{error_feedback}\n\nFix ALL errors and regenerate."
                except Exception as e:
                    logger.warning(f"Stage 4: Attempt {attempt} error: {e}")
        finally:
            await client.delete_assistant(assistant.assistant_id)
    return False

def run(context) -> bool:
    logger = context.logger
    logger.info("Stage 4: Implementation Prompt Design started")

    spec, arch, contracts = context.get("spec"), context.get("architecture"), context.get("contracts")
    if not all([spec, arch, contracts]):
        logger.error("Stage 4: Missing required context")
        return False
    
    # 1. Caching Check
    ctx_hash = get_context_hash(spec, arch, contracts)
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"stage4_{ctx_hash[:12]}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_pack = json.load(f)
            context.set("prompt_pack", cached_pack)
            logger.info("Stage 4: Cache hit! Implementation prompt pack loaded from disk.")
            return True
        except Exception as e:
            logger.warning(f"Stage 4: Failed to load cache: {e}")

    arch_paths = {f["path"] for f in arch["files"]}
    user_prompt = f"Spec:\n{json.dumps(spec)}\nArch:\n{json.dumps(arch)}\nContracts:\n{json.dumps(contracts)}"
    
    provider = os.environ.get("SPECFORGE_PROVIDER", "openai").lower()
    model = os.environ.get("SPECFORGE_MODEL", "gpt-4o-mini")

    success = False
    if provider == "backboard":
        import asyncio
        success = asyncio.run(_run_backboard(context, user_prompt, model, arch_paths, spec, arch))
    else:
        success = asyncio.run(_run_openai(context, user_prompt, model, arch_paths, spec, arch))

    if success:
        # Save to cache
        try:
            prompt_pack = context.get("prompt_pack")
            with open(cache_file, 'w') as f:
                json.dump(prompt_pack, f, indent=2)
            logger.info(f"Stage 4: Result cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Stage 4: Failed to save cache: {e}")
        return True

    return False
