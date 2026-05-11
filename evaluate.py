"""
evaluate.py — Evaluation harness for the SHL Recommender RAG pipeline.

Measures (see ``DEFAULT_SCORING_RUBRIC`` and ``data/evaluation_rubric.json``):
  • Hard evals: ChatResponse schema, SHL/catalog URL groundedness, 8-turn cap
  • Recommendation relevance: Recall@K vs ``expected_assessments`` (endpoint output)
  • Retrieval quality (offline): same Recall@K on hybrid FAISS ``retrieve()`` using trace ``facts``
  • Groundedness: canonical catalog URL match + share of picks that appear in top-20 retrieval
  • Behavior probes + composite effectiveness score

Usage:
    python evaluate.py --init                           # bootstrap dirs + sample traces + rubric snapshot
    python evaluate.py --base-url http://127.0.0.1:8000 # requires running API + traces in data/traces/

Trace JSON schema (minimal):
    {
      "persona": "<optional>",
      "facts": { ... }   # mirrors agent extraction fields for offline retrieve(); optional but recommended
      "conversation": [{"role":"user"|"assistant","content":"..."}],
      "expected_assessments": ["Catalog Name Exact", ...],
      "expectations": {                         # optional behavior checks
          "first_turn_recommendations_empty": true,
          "first_turn_must_refuse_or_empty_recs": true
      }
    }
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

from catalog import build_catalog_index, load_catalog
from retriever import SHLRetriever
from schemas import ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACES_DIR = PROJECT_ROOT / "data" / "traces"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval_results.json"
DEFAULT_RUBRIC_OUT = PROJECT_ROOT / "data" / "evaluation_rubric.json"
# Optional local-only doc; deploy omits this — rubric still works from the built-in default below.
INSTRUCTIONS_JSON = PROJECT_ROOT / "instructions.json"

# Bundled rubric (deploy-safe). Used when ``data/evaluation_rubric.json`` is missing and as fallback.
DEFAULT_SCORING_RUBRIC: dict[str, Any] = {
    "hard_evals": {
        "weight": "Must pass — failure here means zero score",
        "checks": [
            "Every response matches the {reply, recommendations, end_of_conversation} schema exactly",
            "Every URL in recommendations comes from the SHL catalog",
            "Conversation never exceeds 8 turns",
        ],
    },
    "recall_at_10": {
        "weight": "Primary quality metric",
        "definition": (
            "Recall@10 = (relevant assessments in top 10 recommendations) / "
            "(total relevant assessments for query). Averaged across all traces including holdout set."
        ),
    },
    "behavior_probes": {
        "weight": "Secondary quality metric",
        "examples": [
            "Agent refuses off-topic queries",
            "Agent does not recommend on turn 1 for a vague query",
            "Agent honors mid-conversation refinements",
            "Hallucination rate across all turns",
        ],
    },
}


# ---------------------------------------------------------------------------
# Layout + rubric (auto-created resources)
# ---------------------------------------------------------------------------


def _rubric_from_optional_instructions() -> Optional[dict[str, Any]]:
    """If ``instructions.json`` exists (local dev), return its scoring_rubric; else None."""
    if not INSTRUCTIONS_JSON.exists():
        return None
    try:
        with open(INSTRUCTIONS_JSON, encoding="utf-8") as f:
            instr = json.load(f)
        alt = instr.get("scoring_rubric")
        return alt if isinstance(alt, dict) and alt else None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read instructions.json for rubric override: %s", e)
        return None


def sync_evaluation_rubric(force: bool = False) -> dict[str, Any]:
    """
    Resolve scoring rubric: optional ``instructions.json`` override, else built-in default.
    Writes ``data/evaluation_rubric.json`` when missing or when ``force`` is True.
    """
    DEFAULT_RUBRIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    rubric = copy.deepcopy(DEFAULT_SCORING_RUBRIC)
    alt = _rubric_from_optional_instructions()
    if alt is not None:
        rubric = alt
        logger.info("Using scoring_rubric from instructions.json (local override)")
    else:
        logger.debug("Using built-in DEFAULT_SCORING_RUBRIC (no instructions.json)")

    if force or not DEFAULT_RUBRIC_OUT.exists():
        with open(DEFAULT_RUBRIC_OUT, "w", encoding="utf-8") as f:
            json.dump(rubric, f, indent=2)
            f.write("\n")
        logger.info("Wrote scoring rubric to %s", DEFAULT_RUBRIC_OUT)
    return rubric


def load_rubric_snapshot() -> dict[str, Any]:
    """Load rubric for eval reports; ensures file exists from built-in default if needed."""
    if DEFAULT_RUBRIC_OUT.exists():
        try:
            with open(DEFAULT_RUBRIC_OUT, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Corrupt or empty %s — regenerating: %s", DEFAULT_RUBRIC_OUT, e)
    return sync_evaluation_rubric(force=True)


def ensure_evaluation_layout(sync_rubric: bool = False) -> None:
    DEFAULT_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if sync_rubric:
        sync_evaluation_rubric(force=False)


def _write_trace(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def bootstrap_sample_traces() -> None:
    """Populate data/traces with small examples exercising probes + offline retrieval."""
    ensure_evaluation_layout(sync_rubric=True)
    sync_evaluation_rubric(force=True)

    trace_dir = DEFAULT_TRACES_DIR

    full_ctx = (
        "We are hiring fifty entry-level contact centre agents in the UK handling inbound calls—English only. "
        "We need screening under forty-five minutes total per candidate where possible and remote delivery. "
        "We especially want simulations or realistic phone-style assessments plus customer-service orientation "
        "and aptitude for multitasking. What SHL assessments should we use?"
    )
    facts_full = {
        "role": "Contact centre agent",
        "seniority": "Entry-Level",
        "skills": [],
        "test_categories": ["Simulations"],
        "assessment_goals": "Screen CS aptitude for high-volume inbound hires",
        "remote_required": True,
        "max_duration_minutes": 45,
        "languages": ["English"],
        "language_scope": "english_acceptable",
        "volume_or_scale": "50 hires",
        "channel_or_work_context": "Inbound calls / UK contact centre",
        "context_summary": "UK entry-level contact centre inbound English remote max 45 min screening",
    }
    _write_trace(
        trace_dir / "01_full_context_contact_centre.json",
        {
            "id": "01_full_context_contact_centre",
            "persona": "High-volume recruiter",
            "facts": facts_full,
            "conversation": [{"role": "user", "content": full_ctx}],
            "expected_assessments": [
                "Customer Service Phone Simulation",
            ],
            "expectations": {},
        },
    )

    _write_trace(
        trace_dir / "02_probe_vague_first_turn.json",
        {
            "id": "02_probe_vague_first_turn",
            "persona": "Vague shopper",
            "facts": {},
            "conversation": [{"role": "user", "content": "I need an assessment."}],
            "expected_assessments": [],
            "expectations": {"first_turn_recommendations_empty": True},
        },
    )

    _write_trace(
        trace_dir / "03_probe_off_topic.json",
        {
            "id": "03_probe_off_topic",
            "persona": "Off-topic tester",
            "facts": {},
            "conversation": [
                {
                    "role": "user",
                    "content": "Who won an imaginary FIFA World Cup in 2222 and what is 9+9?",
                },
            ],
            "expected_assessments": [],
            "expectations": {"first_turn_must_refuse_or_empty_recs": True},
        },
    )

    schema_hint = {
        "_comment": (
            "SHL-provided traces follow the same shape: persona, facts, conversation (user turns replayed), "
            "expected_assessments; optional expectations for behavior probes."
        ),
        "trace_shape": {
            "persona": "<string>",
            "facts": "<object — extraction-shaped fields used for offline retrieval metrics>",
            "conversation": [{"role": "user|assistant", "content": "..."}],
            "expected_assessments": ["exact catalog assessment name"],
            "expectations": {
                "first_turn_recommendations_empty": "<bool>",
                "first_turn_must_refuse_or_empty_recs": "<bool>",
            },
        },
        "metrics_reference_file": "data/evaluation_rubric.json (built-in default; optional instructions.json override locally)",
    }
    _write_trace(trace_dir / "_trace_schema.example.json", schema_hint)

    logger.info(f"Sample traces written under {trace_dir.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    if not relevant:
        return 1.0
    top_k = [r.lower() for r in recommended[:k]]
    rel = [r.lower() for r in relevant]
    return sum(1 for r in rel if r in top_k) / len(rel)


def _filters_from_extracted(extracted: dict) -> dict:
    filters: dict = {}
    if extracted.get("remote_required") is True:
        filters["remote"] = True
    if extracted.get("adaptive_required") is True:
        filters["adaptive"] = True
    if extracted.get("max_duration_minutes"):
        filters["max_duration"] = extracted["max_duration_minutes"]
    if extracted.get("test_categories"):
        filters["keys"] = extracted["test_categories"]
    if extracted.get("seniority"):
        filters["job_levels"] = [extracted["seniority"]]
    return filters


def offline_retrieval_recalls_and_names(
    retriever: SHLRetriever,
    facts: dict,
    expected: list[str],
    k_vals: tuple[int, ...] = (10, 20),
) -> tuple[dict[str, float], list[str]]:
    extracted = dict(facts or {})
    context_summary = extracted.get("context_summary") or extracted.get("role") or ""
    filters = _filters_from_extracted(extracted)
    top_n = max(k_vals)
    candidates = retriever.retrieve(
        context=context_summary,
        extracted=extracted,
        filters=filters,
        top_n=top_n,
    )
    names = [c["name"] for c in candidates]
    out: dict[str, float] = {}
    for k in k_vals:
        out[f"retrieval_recall_at_{k}"] = recall_at_k(names, expected, k=k)
    return out, names


def url_groundedness(
    recs: list[dict],
    name_to_entry: dict[str, dict],
) -> tuple[float, int, list[str]]:
    """
    Fraction of recommendations whose URL exactly matches catalog canonical link (case/name match).
    """
    if not recs:
        return 1.0, 0, []
    errs: list[str] = []
    hits = 0
    for r in recs:
        nm = (r.get("name") or "").strip()
        url = (r.get("url") or "").strip()
        row = name_to_entry.get(nm.lower())
        if not row:
            errs.append(f"No catalog row for recommended name '{nm}'")
            continue
        if row.get("link", "").rstrip("/") != url.rstrip("/"):
            errs.append(f"URL mismatch for '{nm}' — expected catalog link.")
            continue
        if "shl.com" not in url.lower():
            errs.append(f"Non-SHL URL for '{nm}'.")
            continue
        hits += 1
    return hits / len(recs), hits, errs


def recommendations_aligned_with_retrieval(
    rec_names: list[str],
    retrieval_top_names_lower: set[str],
) -> float:
    """Share of picks that appear in top-20 hybrid retrieval pool (offline)."""
    if not rec_names:
        return 1.0
    ok = sum(1 for n in rec_names if n.lower() in retrieval_top_names_lower)
    return ok / len(rec_names)


def composite_effectiveness(
    rec_recall: float,
    retr_recall: float,
    align: float,
) -> float:
    """Simple interpretable composite (instructions leave weighting implicit)."""
    return float(np.clip(0.45 * rec_recall + 0.35 * retr_recall + 0.20 * align, 0.0, 1.0))


# ---------------------------------------------------------------------------
# HTTP replay (live API)
# ---------------------------------------------------------------------------


def wait_for_health(base_url: str, max_wait: int = 120) -> bool:
    logger.info(f"Waiting for GET {base_url}/health …")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                logger.info("Service is healthy.")
                return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("Service did not become healthy in time.")
    return False


def replay_trace_with_api(
    trace: dict,
    base_url: str,
    timeout: int = 120,
) -> dict[str, Any]:
    history: list[dict] = []
    final_recs: list[dict] = []
    errors: list[str] = []
    schema_ok_turns = 0
    total_turns = 0

    probes = {
        "schema_compliant_all_turns": True,
        "turn_cap_ok": True,
        "urls_shl_only": True,
        "urls_catalog_canonical_ok": True,
        "first_turn_empty_recommendations": None,
        "first_turn_refusal_or_empty": None,
    }

    scripted = [m for m in trace.get("conversation", []) if m["role"] == "user"]

    expectations = trace.get("expectations") or {}

    for turn_idx, user_turn in enumerate(scripted):
        if turn_idx >= 8:
            probes["turn_cap_ok"] = False
            errors.append(f"Exceeded 8 scripted user turns (stopped at simulated turn index {turn_idx}).")
            break

        history.append({"role": "user", "content": user_turn["content"]})
        payload = {"messages": history}
        total_turns += 1

        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            errors.append(f"Turn {turn_idx + 1}: timeout after {timeout}s")
            probes["schema_compliant_all_turns"] = False
            break
        except Exception as exc:
            errors.append(f"Turn {turn_idx + 1}: {exc}")
            probes["schema_compliant_all_turns"] = False
            break

        # Hard eval: pydantic validates API contract (reply / recommendations / eoc)
        try:
            ChatResponse.model_validate(data)
            schema_ok_turns += 1
        except Exception as exc:
            probes["schema_compliant_all_turns"] = False
            errors.append(f"Turn {turn_idx + 1}: schema drift — {exc}")

        reply = data.get("reply") or ""

        # First-turn behavior probes (rubric "behavior_probes")
        if turn_idx == 0:
            if expectations.get("first_turn_recommendations_empty"):
                probes["first_turn_empty_recommendations"] = (
                    len(data.get("recommendations", []) or []) == 0
                )
            refuse_tokens = ("only able to help", "can't assist", "cannot assist",
                            "outside my scope", "not able to help")
            if expectations.get("first_turn_must_refuse_or_empty_recs"):
                rec_empty = len(data.get("recommendations", []) or []) == 0
                textual_refusal = any(t in reply.lower() for t in refuse_tokens)
                probes["first_turn_refusal_or_empty"] = rec_empty or textual_refusal

        for rec in data.get("recommendations", []) or []:
            u = rec.get("url", "") or ""
            if u and "shl.com" not in u.lower():
                probes["urls_shl_only"] = False
                errors.append(f"Turn {turn_idx + 1}: suspicious host in URL '{u}'")

        if data.get("recommendations"):
            final_recs = list(data["recommendations"])

        history.append({"role": "assistant", "content": reply})

        if data.get("end_of_conversation"):
            break

    return {
        "persona": trace.get("persona", "unknown"),
        "final_recommendations_raw": final_recs,
        "errors": errors,
        "total_user_turns": total_turns,
        "schema_ok_turn_fraction": schema_ok_turns / max(total_turns, 1),
        "probes": probes,
        "hard_eval_aggregate": probes["schema_compliant_all_turns"]
        and probes["turn_cap_ok"]
        and probes["urls_shl_only"],
    }


# ---------------------------------------------------------------------------
# Aggregate run
# ---------------------------------------------------------------------------


def run_evaluation(
    base_url: str,
    traces_dir: Path,
    output_path: Path,
    retrieval: SHLRetriever,
    catalog: list[dict],
    name_index: dict[str, dict],
) -> dict[str, Any]:
    trace_files = sorted(traces_dir.glob("*.json"))
    trace_files = [p for p in trace_files if not p.name.startswith("_")]

    rubric_snapshot = load_rubric_snapshot()

    results: list[dict] = []

    name_to_entry = {e["name"].lower(): e for e in catalog}

    eff_scores: list[float] = []

    agg = {
        "mean_recommendation_recall_at_10": [],
        "mean_retrieval_recall_at_10": [],
        "mean_retrieval_recall_at_20": [],
        "mean_url_grounding_rate": [],
        "mean_retrieval_alignment": [],
        "mean_effectiveness": [],
        "hard_eval_trace_pass": [],
    }

    for tf in trace_files:
        with open(tf, encoding="utf-8") as f:
            trace = json.load(f)

        logger.info(f"Replaying trace {tf.name} — persona={trace.get('persona', '?')!r}")

        api_out = replay_trace_with_api(trace, base_url)

        facts = trace.get("facts") or {}
        expected = trace.get("expected_assessments") or []

        rec_names = [
            str(r["name"]).strip()
            for r in api_out["final_recommendations_raw"]
            if isinstance(r, dict)
        ]

        rr, retr_names_list = offline_retrieval_recalls_and_names(
            retrieval, facts, expected, k_vals=(10, 20)
        )
        retr_names_lower = {n.lower() for n in retr_names_list[:20]}

        grounding_rate, grounding_hits, g_err = url_groundedness(
            api_out["final_recommendations_raw"],
            name_to_entry,
        )
        urls_canonical_ok = grounding_rate >= 1.0 and api_out["probes"]["urls_shl_only"]
        api_out["probes"]["urls_catalog_canonical_ok"] = urls_canonical_ok
        api_out.setdefault("errors", []).extend(g_err)

        recall_final = recall_at_k(rec_names, expected, k=10)
        align_pool = recommendations_aligned_with_retrieval(rec_names, retr_names_lower)
        effectiveness = composite_effectiveness(
            recall_final,
            rr.get("retrieval_recall_at_10", 0.0),
            align_pool,
        )
        eff_scores.append(effectiveness)

        hard_pass = (
            api_out["hard_eval_aggregate"]
            and urls_canonical_ok
            and api_out["probes"]["schema_compliant_all_turns"]
        )

        expectations = trace.get("expectations") or {}
        if expectations.get("first_turn_recommendations_empty") is not None:
            ok = api_out["probes"].get("first_turn_empty_recommendations") is True
            hard_pass = hard_pass and ok
            if not ok:
                api_out["errors"].append("Probe failed: vague first-turn should return zero recommendations.")
        if expectations.get("first_turn_must_refuse_or_empty_recs") is not None:
            ok = api_out["probes"].get("first_turn_refusal_or_empty") is True
            hard_pass = hard_pass and ok
            if not ok:
                api_out["errors"].append("Probe failed: off-topic first turn should refuse or emit no recs.")

        row = {
            "trace_file": tf.name,
            "expected_assessments": expected,
            "recommendation_recall_at_10": recall_final,
            "retrieval_recall_at_10": rr.get("retrieval_recall_at_10", 0.0),
            "retrieval_recall_at_20": rr.get("retrieval_recall_at_20", 0.0),
            "url_grounding_rate": grounding_rate,
            "recommendation_retrieval_alignment": align_pool,
            "effectiveness_composite": effectiveness,
            "hard_eval_pass": hard_pass,
            "final_recommendation_names": rec_names,
            **api_out,
        }
        results.append(row)

        agg["mean_recommendation_recall_at_10"].append(recall_final)
        agg["mean_retrieval_recall_at_10"].append(rr.get("retrieval_recall_at_10", 0.0))
        agg["mean_retrieval_recall_at_20"].append(rr.get("retrieval_recall_at_20", 0.0))
        agg["mean_url_grounding_rate"].append(grounding_rate)
        agg["mean_retrieval_alignment"].append(align_pool)
        agg["mean_effectiveness"].append(effectiveness)
        agg["hard_eval_trace_pass"].append(1.0 if hard_pass else 0.0)

        logger.info(
            f"  rec@10={recall_final:.2f} retr@10={rr.get('retrieval_recall_at_10', 0):.2f} "
            f"align={align_pool:.2f} eff={effectiveness:.2f} hard_ok={hard_pass}"
        )
        if api_out["errors"]:
            for e in api_out["errors"][:5]:
                logger.warning(f"    ⚠ {e}")

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "rubric_snapshot": rubric_snapshot,
        "total_traces": len(results),
        "aggregate": {
            "mean_recommendation_recall_at_10": round(_mean(agg["mean_recommendation_recall_at_10"]), 4),
            "mean_retrieval_recall_at_10": round(_mean(agg["mean_retrieval_recall_at_10"]), 4),
            "mean_retrieval_recall_at_20": round(_mean(agg["mean_retrieval_recall_at_20"]), 4),
            "mean_url_grounding_rate": round(_mean(agg["mean_url_grounding_rate"]), 4),
            "mean_recommendation_retrieval_alignment": round(_mean(agg["mean_retrieval_alignment"]), 4),
            "mean_effectiveness_composite": round(_mean(agg["mean_effectiveness"]), 4),
            "hard_eval_pass_rate": round(_mean(agg["hard_eval_trace_pass"]), 4),
        },
        "per_trace": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 56)
    print("EVALUATION SUMMARY (see data/evaluation_rubric.json)")
    print(f"  Traces evaluated : {len(results)}")
    ag = summary["aggregate"]
    print(f"  Mean rec@10 (API)         : {ag['mean_recommendation_recall_at_10']:.4f}")
    print(f"  Mean retr@10 (offline)    : {ag['mean_retrieval_recall_at_10']:.4f}")
    print(f"  Mean retr@20 (offline)    : {ag['mean_retrieval_recall_at_20']:.4f}")
    print(f"  Mean URL grounding        : {ag['mean_url_grounding_rate']:.4f}")
    print(f"  Mean picks-in-top20-retr  : {ag['mean_recommendation_retrieval_alignment']:.4f}")
    print(f"  Mean effectiveness        : {ag['mean_effectiveness_composite']:.4f}")
    print(f"  Hard-eval pass rate       : {ag['hard_eval_pass_rate']:.0%}")
    print(f"\nFull JSON → {output_path}")
    print("=" * 56 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SHL RAG recommender (API + offline retrieval)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create data/traces layout, sample traces, schema example, and data/evaluation_rubric.json (built-in rubric)",
    )
    parser.add_argument(
        "--sync-rubric",
        action="store_true",
        help="Force-rewrite data/evaluation_rubric.json (instructions.json if present, else built-in default)",
    )
    args = parser.parse_args()

    if args.init:
        bootstrap_sample_traces()
        print("Bootstrap complete. Add more JSON traces under data/traces/ then run without --init.")
        return

    os.chdir(PROJECT_ROOT)
    ensure_evaluation_layout(sync_rubric=args.sync_rubric)
    if args.sync_rubric:
        sync_evaluation_rubric(force=True)
    else:
        sync_evaluation_rubric(force=False)

    if not wait_for_health(args.base_url):
        sys.exit(1)

    catalog_path = PROJECT_ROOT / "data" / "catalog.json"
    if not catalog_path.exists():
        logger.error(f"Missing {catalog_path}")
        sys.exit(1)

    catalog = load_catalog(str(catalog_path))
    name_index = build_catalog_index(catalog)
    retriever = SHLRetriever()
    retriever.build_index(catalog=catalog, name_index=name_index, force_rebuild=False)

    if not args.traces_dir.exists():
        logger.error(f"Traces directory missing: {args.traces_dir}. Run: python evaluate.py --init")
        sys.exit(1)

    trace_files = [p for p in sorted(args.traces_dir.glob("*.json")) if not p.name.startswith("_")]
    if not trace_files:
        logger.error(
            f"No trace JSON files in {args.traces_dir}. Run: python evaluate.py --init "
            "or install SHL traces from the assignment zip."
        )
        sys.exit(1)

    run_evaluation(
        base_url=args.base_url,
        traces_dir=args.traces_dir,
        output_path=args.output,
        retrieval=retriever,
        catalog=catalog,
        name_index=name_index,
    )


if __name__ == "__main__":
    main()
