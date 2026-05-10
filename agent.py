"""
agent.py
Core agent logic.
Handles intent detection, context extraction, retrieval orchestration,
LLM calls, and response assembly.

Design principle: ONE LLM call per /chat request (for the final reply).
A second smaller call handles extraction. Reranking is done in the same
final call. This keeps total latency well within the 30s timeout.
"""

import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from catalog import (
    format_recommendation_test_type,
    get_catalog_summary,
)
from prompts import (
    SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    RERANK_PROMPT,
    COMPARE_PROMPT,
)
from retriever import SHLRetriever
from schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client (Gemini 3.1 Flash Lite via OpenAI-compatible SDK)
# ---------------------------------------------------------------------------
#
# Google's quota/RPC errors sometimes mention older Flash SKUs inside
# quotaDimensions—that reflects how the API meters usage, not necessarily
# the literal model id sent in the request.


def get_gemini_model_id() -> str:
    """Model id for chat completions; override via GEMINI_MODEL in `env`."""
    load_dotenv("env", override=True)
    v = (os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite").strip()
    return v or "gemini-3.1-flash-lite"


def _chat_once(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Single Gemini completion call.
    """
    model_id = get_gemini_model_id()
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def _make_client() -> OpenAI:
    # Use only the local `env` file for key loading.
    load_dotenv("env", override=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set in env file.")
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=0,
        timeout=20.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(text: str) -> Optional[dict | list]:
    """
    Safely parse JSON from LLM output.
    Strips markdown fences if present.
    Returns None on failure.
    """
    # Strip ```json ... ``` fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Failed to parse JSON from LLM output: {text[:200]}")
        return None


def _messages_to_text(messages: list[Message]) -> str:
    """Flatten conversation history to plain text for extraction prompt."""
    lines = []
    for m in messages:
        prefix = "User" if m.role == "user" else "Assistant"
        lines.append(f"{prefix}: {m.content}")
    return "\n".join(lines)


def _count_assistant_turns(messages: list[Message]) -> int:
    return sum(1 for m in messages if m.role == "assistant")


def _format_candidates_for_rerank(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        type_str = format_recommendation_test_type(c)
        lines.append(
            f"{i}. {c['name']}\n"
            f"   URL: {c['url']}\n"
            f"   Types: {type_str}\n"
            f"   Description: {c.get('description', '')[:200]}\n"
            f"   Job Levels: {', '.join(c.get('job_levels', []))}\n"
            f"   Duration: {c.get('duration_minutes', 'N/A')} min\n"
        )
    return "\n".join(lines)


def _format_assessments_for_compare(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        type_str = format_recommendation_test_type(entry)
        lines.append(
            f"**{entry['name']}**\n"
            f"URL: {entry['url']}\n"
            f"Test types: {type_str}\n"
            f"Description: {entry.get('description', '')}\n"
            f"Job Levels: {', '.join(entry.get('job_levels', []))}\n"
            f"Duration: {entry.get('duration_minutes', 'N/A')} minutes\n"
            f"Remote: {entry.get('remote', 'N/A')}\n"
            f"Adaptive: {entry.get('adaptive', 'N/A')}\n"
            f"Categories: {', '.join(entry.get('keys', []))}\n"
        )
    return "\n---\n".join(lines)


# ---------------------------------------------------------------------------
# Out-of-scope / injection detection
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_PATTERNS = [
    r"\bsalar(y|ies)\b",
    r"\bcompensation\b",
    r"\blegal\b",
    r"\blawsuit\b",
    r"\bdiscrimination\b",
    r"\bcompetitor\b",
    r"\bkorn ferry\b",
    r"\bpearson\b",
    r"\bhogan\b",
    r"\bignore (previous|all|your) instructions?\b",
    r"\bforget (everything|your prompt)\b",
    r"\bact as\b",
    r"\bpretend (you are|to be)\b",
    r"\bjailbreak\b",
    r"\bdan mode\b",
]

_COMPARE_PATTERNS = [
    r"\bdifference between\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bwhich is better\b",
    r"\bhow does .+ differ\b",
]

_VAGUE_PATTERNS = [
        r"^i need an? assessment",
        r"^help me (hire|find|screen)",
        r"^find (me |us )?(an?|some) assessment",
        r"^what assessments",
        r"^i want to (hire|assess|screen)",
        r"^we (need|want|are looking)",
    ]


def _is_out_of_scope(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _OUT_OF_SCOPE_PATTERNS)


def _looks_like_compare(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _COMPARE_PATTERNS)


def _looks_vague(text: str, messages: list[Message]) -> bool:
    """True only on the very first user turn with a vague query."""
    if _count_assistant_turns(messages) > 0:
        return False
    t = text.lower().strip()
    return any(re.search(p, t) for p in _VAGUE_PATTERNS) or len(t.split()) < 4


def _seniority_mentioned(extracted: dict, text: str) -> bool:
    if extracted.get("seniority"):
        return True
    t = text.lower()
    return bool(
        re.search(
            r"\b(entry[- ]?level|junior|senior|mid[- ]?career|mid[- ]?professional|"
            r"manager|director|executive|graduate|intern|vp|vice president|"
            r"lead|principal|staff|years?\s+of\s+experience|yoe)\b",
            t,
        )
    )


def _combined_user_text(messages: list[Message]) -> str:
    """All user utterances in order — used for slot checks across the whole dialogue."""
    return "\n".join(m.content for m in messages if m.role == "user")


def _role_mentioned(extracted: dict, text: str) -> bool:
    if (extracted.get("role") or "").strip():
        return True
    t = text.lower()
    return bool(
        re.search(
            r"\b(administrative|admin|assistant|developer|engineer|analyst|"
            r"manager|director|sales|recruit|hire|role|position|job|agent|representative|"
            r"associate|operator|officer|nurse|teacher|clerk|technician|"
            r"contact\s*cent(re|er)|call\s*cent(re|er)|help\s*desk|support|"
            r"customer\s*service|service\s*desk)\b",
            t,
        )
    )


def _focus_mentioned(extracted: dict, text: str) -> bool:
    skills = extracted.get("skills") or []
    cats = extracted.get("test_categories") or []
    goals = (extracted.get("assessment_goals") or "").strip()
    if skills or cats or goals:
        return True
    t = text.lower()
    return bool(
        re.search(
            r"\b(excel|word|outlook|office|powerpoint|spreadsheet|typing|data entry|"
            r"python|java|sql|coding|programming|personality|opq|numerical|verbal|"
            r"reasoning|aptitude|cognitive|sjt|situational|leadership|multitask|"
            r"customer\s*service|inbound|outbound|retail|screen|assess|test|skill|"
            r"screening|sift|volume\s*hire|high[\s-]*volume|"
            r"simulation|in[- ]?basket|day[- ]?in[- ]?life|work\s*sample)\b",
            t,
        )
    )


def _language_slot_satisfied(extracted: dict, text: str) -> bool:
    langs = extracted.get("languages") or []
    if langs:
        return True
    scope = (extracted.get("language_scope") or "").lower()
    if scope in ("english_acceptable", "english_only", "languages_specified", "any"):
        return True
    t = text.lower()
    if re.search(
        r"\b(english[- ]only|english is fine|just english|monolingual|"
        r"us english|uk english)\b",
        t,
    ):
        return True
    if re.search(
        r"\b(spanish|french|german|mandarin|cantonese|hindi|portuguese|dutch|"
        r"japanese|korean|arabic|italian|polish|turkish|russian|bilingual|"
        r"multilingual|must speak|language requirement)\b",
        t,
    ):
        return True
    return False


def _logistics_slot_satisfied(extracted: dict, text: str) -> bool:
    t = text.lower()
    if extracted.get("max_duration_minutes") is not None:
        time_ok = True
    elif extracted.get("time_budget_flexible") is True:
        time_ok = True
    elif re.search(
        r"\b(\d+\s*-\s*\d+\s*(min|minutes)|\d+\s*(min|minutes)|under\s*\d+|"
        r"max(imum)?\s*(of\s*)?\d+|no more than\s*\d+|"
        r"time\s*(budget|limit|cap)|\b\d+\s*min\b)\b",
        t,
    ):
        time_ok = True
    elif re.search(
        r"\b(unlimited|no limit|flexible on time|time is flexible|"
        r"as long as needed|whatever fits)\b",
        t,
    ):
        time_ok = True
    else:
        time_ok = False

    if extracted.get("remote_required") is True or extracted.get("remote_required") is False:
        remote_ok = True
    elif extracted.get("remote_preference_clear") is True:
        remote_ok = True
    elif re.search(
        r"\b(remote|online only|wfh|work from home|virtual|at home|"
        r"on-?site|in-?office|in person|hybrid|proctor|test centre|test center)\b",
        t,
    ):
        remote_ok = True
    else:
        remote_ok = False

    return time_ok and remote_ok


def _missing_recommendation_slot(extracted: dict, user_text: str) -> Optional[str]:
    """
    First missing required slot for a defensible shortlist, in priority order.
    None = all required slots satisfied.
    """
    if not _role_mentioned(extracted, user_text):
        return "role"
    if not _seniority_mentioned(extracted, user_text):
        return "seniority"
    if not _focus_mentioned(extracted, user_text):
        return "focus"
    if not _language_slot_satisfied(extracted, user_text):
        return "language"
    if not _logistics_slot_satisfied(extracted, user_text):
        return "logistics"
    return None


def _slot_clarify_reply(slot: str) -> str:
    """One focused question per missing slot."""
    if slot == "role":
        return "What is the exact job title or function you are hiring for?"
    if slot == "seniority":
        return (
            "What seniority level or career band is this role "
            "(for example entry-level, mid-career, or manager)?"
        )
    if slot == "focus":
        return (
            "What should we prioritize measuring for this hire "
            "(for example customer-service fit, personality, cognitive ability, or specific tools)?"
        )
    if slot == "language":
        return (
            "Which languages must the assessments support for your candidates "
            "(or is English-only acceptable)?"
        )
    if slot == "logistics":
        return (
            "What is the maximum testing time per candidate you can allow "
            "(approximate minutes), and do you need fully remote delivery or on-site testing?"
        )
    return "Could you share a bit more about the role and what you want to assess?"


def _ready_for_recommendations(extracted: dict, user_text: str) -> bool:
    return _missing_recommendation_slot(extracted, user_text) is None


def normalize_extracted(extracted: dict, user_text: str) -> dict:
    """
    Merge lightweight regex hints when the LLM left fields null.
    Mutates a copy of extracted.
    """
    out = dict(extracted)
    t = user_text.lower()

    if not (out.get("role") or "").strip():
        m = re.search(
            r"\b(screening|hiring|recruiting)\s+(\d+)\s+([a-z\s\-]+?)(?:\.|,|\s+for|\s+in\b)",
            t,
            re.I,
        )
        if m:
            out["role"] = m.group(3).strip()

    if out.get("seniority") is None:
        sm = re.search(
            r"\b(entry[- ]?level|graduate|intern|junior|senior|mid[- ]?professional|"
            r"manager|director|executive)\b",
            t,
        )
        if sm:
            raw = sm.group(1).lower()
            if "entry" in raw:
                out["seniority"] = "Entry-Level"
            elif "mid" in raw and "professional" in raw:
                out["seniority"] = "Mid-Professional"
            elif "graduate" in raw:
                out["seniority"] = "Graduate"
            elif "intern" in raw:
                out["seniority"] = "Entry-Level"
            else:
                out["seniority"] = sm.group(1).title()

    dur_m = re.search(
        r"\b(?:max|maximum|under|no more than|at most)\s*(\d+)\s*(?:min|minutes?)\b",
        t,
    )
    if dur_m and out.get("max_duration_minutes") is None:
        try:
            out["max_duration_minutes"] = int(dur_m.group(1))
        except ValueError:
            pass
    if out.get("max_duration_minutes") is None:
        dur_simple = re.search(
            r"\b(\d+)\s*(?:min|minutes?)\b",
            t,
        )
        if dur_simple:
            try:
                n = int(dur_simple.group(1))
                if n <= 240:
                    out["max_duration_minutes"] = n
            except ValueError:
                pass
    if re.search(r"\b(unlimited|no limit|flexible on time)\b", t):
        out["time_budget_flexible"] = True

    if out.get("remote_required") is None:
        if re.search(r"\b(remote|online only|wfh|virtual|at home)\b", t):
            out["remote_required"] = True
            out["remote_preference_clear"] = True
        elif re.search(r"\b(on-?site|in-?office|in person)\b", t):
            out["remote_required"] = False
            out["remote_preference_clear"] = True

    if not out.get("languages") and re.search(
        r"\b(english[- ]only|english is fine)\b", t
    ):
        out["languages"] = ["English"]
        out["language_scope"] = "english_acceptable"

    if not (out.get("assessment_goals") or "").strip():
        if re.search(r"\b(customer\s*service|contact\s*cent|call\s*cent|inbound)\b", t):
            out["assessment_goals"] = "Customer service / contact-centre fit and screening"

    # Simulation / work-sample cues the extractor sometimes misses in long, narrative turns
    cats = list(out.get("test_categories") or [])
    cat_lower = {c.lower() for c in cats if isinstance(c, str)}
    if "simulations" not in cat_lower and re.search(
        r"\b(simulations?|simulated|in[- ]?basket|day[- ]?in[- ]?(the[- ]?)?life|"
        r"work[- ]?sample|work[- ]?sample[- ]?style|interactive\s+(job\s+)?exercise|"
        r"role[- ]?play|mock\s+(customer|call|client))\b",
        t,
    ):
        cats.append("Simulations")
        out["test_categories"] = cats

    if not (out.get("volume_or_scale") or "").strip():
        vm = re.search(
            r"\b(high[- ]?volume|mass\s+hire|bulk\s+hire|scale\s+hire)\b",
            t,
        )
        if vm:
            out["volume_or_scale"] = vm.group(0)
        else:
            nm = re.search(
                r"\b(\d{2,5})\s*(?:\+)?\s*(?:hires?|roles?|openings?|candidates?|people|heads)\b",
                t,
            )
            if nm:
                out["volume_or_scale"] = f"{nm.group(1)} hires"

    if not (out.get("channel_or_work_context") or "").strip():
        if re.search(r"\b(inbound|outbound)\b.*\b(call|phone|voice)\b", t) or re.search(
            r"\b(call\s*cent|contact\s*cent|help\s*desk|service\s*desk)\b",
            t,
        ):
            out["channel_or_work_context"] = "Contact centre / phone-based service"

    if not (out.get("context_summary") or "").strip():
        parts = []
        if (out.get("role") or "").strip():
            parts.append((out.get("role") or "").strip())
        if (out.get("seniority") or "").strip():
            parts.append((out.get("seniority") or "").strip())
        if (out.get("assessment_goals") or "").strip():
            parts.append((out.get("assessment_goals") or "").strip())
        if parts:
            out["context_summary"] = "; ".join(parts)

    return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_context(
    messages: list[Message],
    client: OpenAI,
) -> dict:
    """
    Use a lightweight LLM call to extract structured fields from conversation.
    Returns dict with role, seniority, skills, test_categories, filters, etc.
    Regex normalization fills gaps when the model omits fields.
    """
    conversation_text = _messages_to_text(messages)
    combined_user = _combined_user_text(messages)
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        raw = _chat_once(
            client=client,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = _safe_json(raw)
        if isinstance(parsed, dict):
            return normalize_extracted(parsed, combined_user)
    except Exception as e:
        logger.warning(f"Extraction LLM call failed: {e}")

    return normalize_extracted({}, combined_user)


# ---------------------------------------------------------------------------
# Compare handler
# ---------------------------------------------------------------------------

def handle_compare(
    user_message: str,
    messages: list[Message],
    retriever: SHLRetriever,
    client: OpenAI,
) -> ChatResponse:
    """
    Handle a comparison request.
    Extracts assessment names from the query, fetches from catalog,
    and uses LLM to write a grounded comparison.
    """
    # Try to find mentioned assessment names via sliding windows over
    # punctuation-stripped tokens (e.g. 'Simulation?' → 'Simulation').
    found_entries = []
    words = [w.strip(".,!?;:'\"") for w in user_message.split()]
    words = [w for w in words if w]

    # Collect all potential multi-word names (try 1–4 word windows)
    for size in range(4, 0, -1):
        for i in range(len(words) - size + 1):
            candidate = " ".join(words[i:i + size])
            entry = retriever.resolve_catalog_name(candidate)
            if entry and entry not in found_entries:
                found_entries.append(entry)

    # If exact lookup failed, use vector search on the whole query
    if len(found_entries) < 2:
        vector_hits = retriever.fuzzy_name_search(user_message, top_k=10)
        for hit in vector_hits:
            full_entry = retriever.resolve_catalog_name(hit["name"])
            if full_entry and full_entry not in found_entries:
                found_entries.append(full_entry)

    if len(found_entries) < 2:
        return ChatResponse(
            reply=(
                "I couldn't identify two or more assessments to compare in your message. "
                "Could you please name the assessments you'd like me to compare? "
                "For example: 'Compare OPQ32r and the Global Skills Assessment.'"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    assessments_data = _format_assessments_for_compare(found_entries[:3])
    compare_prompt = COMPARE_PROMPT.format(
        assessments_data=assessments_data,
        user_question=user_message,
    )

    try:
        reply_text = _chat_once(
            client=client,
            max_tokens=600,
            temperature=0.1,
            messages=[{"role": "user", "content": compare_prompt}],
        ) or "Unable to generate comparison."
    except Exception as e:
        logger.error(f"Compare LLM call failed: {e}")
        reply_text = "I encountered an error generating the comparison. Please try again."

    return ChatResponse(
        reply=reply_text,
        recommendations=[],
        end_of_conversation=False,
    )


# ---------------------------------------------------------------------------
# Main agent call
# ---------------------------------------------------------------------------

def build_catalog_context(candidates: list[dict], max_items: int = 30) -> str:
    """
    Format top candidates as context for the system prompt.
    Limits to max_items to avoid exceeding context window.
    """
    lines = []
    for c in candidates[:max_items]:
        type_str = format_recommendation_test_type(c)
        lines.append(
            f"- Name: {c['name']} | URL: {c['url']} | Types: {type_str} "
            f"| Levels: {', '.join(c.get('job_levels', []))} "
            f"| Duration: {c.get('duration_minutes', 'N/A')} min "
            f"| Remote: {c.get('remote', '?')} "
            f"| Description: {c.get('description', '')[:150]}"
        )
    return "\n".join(lines)


def run_agent(
    messages: list[Message],
    retriever: SHLRetriever,
    catalog: list[dict],
    client: Optional[OpenAI] = None,
) -> ChatResponse:
    """
    Main agent entry point.
    Called once per POST /chat request.

    Flow:
    1. Guard: out-of-scope / injection → refuse
    2. Guard: compare intent → handle_compare
    3. Extract structured context from conversation
    4. Retrieve candidates (hybrid)
    5. Build system prompt with catalog context
    6. Call LLM → parse structured response
    7. Validate all URLs against catalog before returning
    """
    if client is None:
        client = _make_client()

    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    turn_count = _count_assistant_turns(messages)

    # -----------------------------------------------------------------------
    # Guard: out-of-scope / prompt injection
    # -----------------------------------------------------------------------
    if _is_out_of_scope(last_user):
        return ChatResponse(
            reply=(
                "I'm only able to help with SHL assessment selection. "
                "I can't assist with that topic. "
                "If you have a hiring role in mind, I'd be happy to recommend relevant assessments."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # -----------------------------------------------------------------------
    # Guard: compare intent
    # -----------------------------------------------------------------------
    if _looks_like_compare(last_user):
        return handle_compare(last_user, messages, retriever, client)

    # -----------------------------------------------------------------------
    # Guard: vague first turn -> single clarifying question
    # -----------------------------------------------------------------------
    if _looks_vague(last_user, messages):
        return ChatResponse(
            reply="What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # -----------------------------------------------------------------------
    # Extract structured context
    # -----------------------------------------------------------------------
    extracted = extract_context(messages, client)
    combined_user_text = _combined_user_text(messages)
    logger.debug(f"Extracted context: {extracted}")

    # Build metadata filters from extracted context
    filters: dict = {}
    if extracted.get("remote_required") is True:
        filters["remote"] = True
    if extracted.get("adaptive_required") is True:
        filters["adaptive"] = True
    if extracted.get("max_duration_minutes"):
        filters["max_duration"] = extracted["max_duration_minutes"]
    if extracted.get("test_categories"):
        filters["keys"] = extracted["test_categories"]

    seniority = extracted.get("seniority")
    if seniority:
        filters["job_levels"] = [seniority]

    # -----------------------------------------------------------------------
    # Retrieve candidates
    # -----------------------------------------------------------------------
    context_summary = extracted.get("context_summary") or last_user
    candidates = retriever.retrieve(
        context=context_summary,
        extracted=extracted,
        filters=filters,
        top_n=20,
    )
    logger.debug(f"Retrieved {len(candidates)} candidates.")

    # -----------------------------------------------------------------------
    # Build system prompt with catalog context
    # -----------------------------------------------------------------------
    catalog_summary = get_catalog_summary(catalog)
    catalog_context = build_catalog_context(candidates)

    system_content = SYSTEM_PROMPT.format(
        catalog_summary=catalog_summary,
        catalog_context=catalog_context if catalog_context else "No matching assessments found.",
    )

    # -----------------------------------------------------------------------
    # Assemble messages for LLM
    # -----------------------------------------------------------------------
    llm_messages = [{"role": "system", "content": system_content}]
    for m in messages:
        llm_messages.append({"role": m.role, "content": m.content})

    # -----------------------------------------------------------------------
    # LLM call
    # -----------------------------------------------------------------------
    try:
        raw_output = _chat_once(
            client=client,
            max_tokens=1000,
            temperature=0.2,
            messages=llm_messages,
        )
    except Exception as e:
        logger.error(f"Main LLM call failed: {e}")
        # Fallback: keep the service useful during transient/quota LLM failures by
        # returning top retrieved catalog matches directly.
        if candidates:
            if not _ready_for_recommendations(extracted, combined_user_text):
                miss = _missing_recommendation_slot(extracted, combined_user_text) or "role"
                return ChatResponse(
                    reply=_slot_clarify_reply(miss),
                    recommendations=[],
                    end_of_conversation=False,
                )
            fallback_recs = [
                Recommendation(
                    name=c["name"],
                    url=c["url"],
                    test_type=format_recommendation_test_type(c),
                )
                for c in candidates[:5]
            ]
            return ChatResponse(
                reply=(
                    "I couldn't reach the language model right now, so I'm sharing "
                    "the best catalog matches from retrieval."
                ),
                recommendations=fallback_recs,
                end_of_conversation=False,
            )
        return ChatResponse(
            reply="I'm experiencing a temporary issue. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    # -----------------------------------------------------------------------
    # Parse LLM response
    # -----------------------------------------------------------------------
    parsed = _safe_json(raw_output)

    if not isinstance(parsed, dict):
        # LLM didn't return valid JSON — return reply as plain text, no recommendations
        logger.warning("LLM returned non-JSON response; using as plain reply.")
        return ChatResponse(
            reply=raw_output[:1000],
            recommendations=[],
            end_of_conversation=False,
        )

    reply_text = parsed.get("reply", "")
    raw_recs = parsed.get("recommendations", [])
    eoc = bool(parsed.get("end_of_conversation", False))

    # -----------------------------------------------------------------------
    # Validate every recommendation URL against catalog
    # (LLM cannot invent URLs — must come from retrieved candidates)
    # -----------------------------------------------------------------------
    # Build a name→candidate lookup from retrieved candidates
    candidate_lookup: dict[str, dict] = {
        c["name"].lower(): c for c in candidates
    }
    # Also allow lookup from full catalog via retriever
    validated_recs: list[Recommendation] = []

    for r in raw_recs[:10]:  # hard cap at 10
        if not isinstance(r, dict):
            continue
        name = r.get("name", "").strip()
        if not name:
            continue

        # Try candidate lookup first (fastest)
        entry = candidate_lookup.get(name.lower())

        # Fall back to full catalog lookup (relaxed titles, e.g. ' (New)')
        if entry is None:
            full = retriever.resolve_catalog_name(name)
            if full:
                entry = full

        if entry is None:
            logger.warning(f"LLM recommended unknown assessment '{name}' — dropped.")
            continue

        validated_recs.append(
            Recommendation(
                name=entry["name"],
                url=entry["url"],          # always from catalog, never from LLM
                test_type=format_recommendation_test_type(entry),
            )
        )

    if validated_recs and not _ready_for_recommendations(extracted, combined_user_text):
        miss = _missing_recommendation_slot(extracted, combined_user_text) or "role"
        return ChatResponse(
            reply=_slot_clarify_reply(miss),
            recommendations=[],
            end_of_conversation=False,
        )

    # -----------------------------------------------------------------------
    # Turn cap safety: force recommendation by turn 6
    # -----------------------------------------------------------------------
    if turn_count >= 5 and not validated_recs and candidates:
        # Agent has been clarifying too long — commit to best candidates
        logger.info("Turn cap approaching — forcing recommendation from top candidates.")
        for c in candidates[:5]:
            validated_recs.append(
                Recommendation(
                    name=c["name"],
                    url=c["url"],
                    test_type=format_recommendation_test_type(c),
                )
            )
        reply_text += (
            "\n\nBased on the information gathered so far, here are my best recommendations."
        )
        eoc = False

    return ChatResponse(
        reply=reply_text,
        recommendations=validated_recs,
        end_of_conversation=eoc,
    )
