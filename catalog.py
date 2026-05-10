"""
catalog.py
Loads, validates, and preprocesses the SHL catalog JSON.
Acts as the single source of truth for all assessment data.
All URLs returned to users must come from this module — never from the LLM.
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Primary test type mapping from SHL 'keys' field
KEYS_TO_TEST_TYPE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Simulations": "S",
}

# Priority order when an assessment has multiple keys (single-letter codes)
KEY_PRIORITY = ["K", "A", "P", "B", "C", "D", "E", "S"]


def _resolve_test_type(keys: list[str]) -> str:
    """Primary test type — used as the single test_type field in API response."""
    mapped = [KEYS_TO_TEST_TYPE[k] for k in keys if k in KEYS_TO_TEST_TYPE]
    for priority_type in KEY_PRIORITY:
        if priority_type in mapped:
            return priority_type
    return "O"

def _resolve_all_test_types(keys: list[str]) -> list[str]:
    """
    All applicable type codes from catalog `keys`, deduped and ordered by KEY_PRIORITY,
    then any extras (e.g. future key mappings).
    """
    codes: set[str] = set()
    for k in keys:
        if k in KEYS_TO_TEST_TYPE:
            codes.add(KEYS_TO_TEST_TYPE[k])
    ordered = [c for c in KEY_PRIORITY if c in codes]
    for c in sorted(codes - set(ordered)):
        ordered.append(c)
    return ordered


def format_recommendation_test_type(entry: dict) -> str:
    """
    API `test_type` field: all SHL type codes that apply to this assessment,
    comma-separated in catalog priority order (matches primary + secondary keys).
    """
    codes = list(entry.get("all_test_types") or [])
    if not codes and entry.get("keys"):
        codes = _resolve_all_test_types(entry["keys"])
    if codes:
        return ", ".join(codes)
    t = entry.get("test_type")
    return str(t) if t else "O"


def _parse_duration_minutes(duration_raw: str) -> Optional[int]:
    """
    Extract integer minutes from strings like:
    'Approximate Completion Time in minutes = 30'
    Returns None if not parseable.
    """
    if not duration_raw:
        return None
    try:
        # Split on '=' and take the last part
        parts = duration_raw.split("=")
        if len(parts) >= 2:
            return int(parts[-1].strip())
    except (ValueError, IndexError):
        pass
    return None


def build_embedding_text(entry: dict) -> str:
    """
    Construct a rich text string per assessment for embedding.
    Concatenates all semantically meaningful fields.
    This is what gets embedded into the FAISS index.
    """
    parts = [
        f"Name: {entry['name']}",
        f"Description: {entry.get('description', '')}",
        f"Test Types: {', '.join(entry.get('keys', []))}",
        f"Job Levels: {entry.get('job_levels_raw', '')}",
        f"Duration: {entry.get('duration_raw', '')}",
        f"Remote Testing: {entry.get('remote', '')}",
        f"Adaptive: {entry.get('adaptive', '')}",
        f"Languages: {', '.join(entry.get('languages', []))}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


def load_catalog(path: str = "data/catalog.json") -> list[dict]:
    """
    Load and validate the catalog JSON.
    - Filters out entries where status != 'ok'
    - Filters out entries missing name or link
    - Enriches each entry with derived fields: test_type, duration_minutes, embedding_text
    Returns a clean list of assessment dicts ready for indexing and lookup.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalog not found at '{path}'. "
            "Place catalog.json inside the data/ directory."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    catalog = []
    skipped = 0

    for entry in raw:
        # Hard filters
        if entry.get("status") != "ok":
            skipped += 1
            continue
        if not entry.get("name") or not entry.get("link"):
            skipped += 1
            continue

        # Enrich
        entry["test_type"] = _resolve_test_type(entry.get("keys", []))
        entry["all_test_types"] = _resolve_all_test_types(entry.get("keys", []))
        entry["duration_minutes"] = _parse_duration_minutes(
            entry.get("duration_raw", "")
        )
        entry["embedding_text"] = build_embedding_text(entry)

        catalog.append(entry)

    logger.info(
        f"Catalog loaded: {len(catalog)} valid entries, {skipped} skipped."
    )
    return catalog


def build_catalog_index(catalog: list[dict]) -> dict[str, dict]:
    """
    Build a fast name→entry lookup dict.
    Keys are lowercased names for case-insensitive matching.
    Used for direct compare lookups without going through FAISS.
    """
    return {entry["name"].lower(): entry for entry in catalog}


def get_entry_by_name(name: str, index: dict[str, dict]) -> Optional[dict]:
    """
    Case-insensitive name lookup.
    Returns the catalog entry or None if not found.
    """
    return index.get(name.lower())


def get_catalog_summary(catalog: list[dict]) -> str:
    """
    Returns a concise summary of the catalog for injection into the system prompt.
    Tells the LLM what test types and job levels are available without
    dumping the full catalog (which would exceed context limits).
    """
    all_keys = set()
    all_job_levels = set()
    for entry in catalog:
        all_keys.update(entry.get("keys", []))
        all_job_levels.update(entry.get("job_levels", []))

    return (
        f"The SHL catalog contains {len(catalog)} Individual Test Solutions.\n"
        f"Available test categories: {', '.join(sorted(all_keys))}.\n"
        f"Available job levels: {', '.join(sorted(all_job_levels))}."
    )
