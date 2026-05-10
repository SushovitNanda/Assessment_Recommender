# SHL RAG Recommender — Design Notes

*Internal design summary (~2 pages). For setup and API details see [README.md](README.md).*

## Architecture choices

- **Stateless HTTP** — Every `/chat` call carries the full transcript. No server session store; simpler deployment.
- **Two LLM calls per recommendation turn** — A dedicated **extraction** pass produces structured slots and query fodder; the **main** call generates strict JSON (`reply` + `recommendations`). Trade-off: extra latency and token cost vs. stuffing everything into one brittle completion.
- **Slot gating instead of a fixed turn count** — Early prototypes used a hard minimum number of user messages; that produced unnecessary clarifications when one long message already contained every constraint. The current design requires **role, seniority, focus (what to measure), language/locale, and logistics (time + remote)** before committing to a shortlist, with regex backfills in `normalize_extracted` when the extractor misses surface cues.
- **URL safety** — Recommended names are validated against the catalog; **URLs always come from catalog entries**, not from the model, so hallucinated links cannot reach the client.
- **Gemini via OpenAI-compatible base URL** — Keeps a single client pattern (`openai` SDK) while targeting Google’s endpoint; model id configurable with `GEMINI_MODEL`.

## Retrieval setup

- **Vectors** — `all-MiniLM-L6-v2` on a concatenated `embedding_text` (name, description, SHL keys, job levels, duration, remote/adaptive flags, languages).
- **Index** — FAISS `IndexFlatIP` on L2-normalized embeddings (cosine similarity). Index and metadata are persisted (`data/faiss.index`, `data/faiss_meta.pkl`) so cold start avoids rebuilding.
- **Query side** — Multi-string expansion from role, skills, test categories, seniority, and assessment goals; union of hits, **soft** metadata filters (if a filter zeroes results, it is relaxed), dedupe, score sort, cap before prompt injection.
- **Compare path** — Sliding n-grams over the user question plus optional vector fallback; **relaxed name resolution** (punctuation, ` (New)` suffixes) so natural questions still resolve to catalog rows.

## Prompt design

- **System prompt** — Grounds the model in the retrieved slice only; defines intents (CLARIFY / RECOMMEND / REFINE / COMPARE), JSON schema, and slot awareness so the model does not “wishlist” products outside context.
- **Extraction prompt** — JSON-only structured fields for filters and retrieval; explicit instructions for **long, non-linear** user messages and SHL taxonomy (including **Simulations**).
- **Compare prompt** — Uses only serialized catalog fields passed in; instruction not to claim a product is missing when it is present in the provided blocks.
- **Rerank prompt** — Optional shortlist ordering by name from candidate text (exact name discipline).

## Evaluation approach

- **Manual / ad hoc:** Exercise `POST /chat` with realistic multi-turn histories; use `print_chat.py` to avoid truncated JSON in shells.
- **Qualitative checklist:** Empty recommendations when clarifying; catalog URLs only in shortlists; compare answers grounded in retrieved catalog blocks.

## What did not work (or needed iteration)

- **Fixed “N user messages before recommend”** — Forced extra turns when a single message already contained all slots; replaced with **slot satisfaction** checks plus richer extraction and regex normalization.
- **Exact string compare lookups** — User phrasing and catalog labels diverge (`(New)`, trailing `?`); fixed with **token cleanup** and `resolve_catalog_name` rather than relying on embedding luck alone.
- **Single retrieval query** — Missed relevant tools when wording differed from catalog prose; **multi-query expansion** improved recall subjectively.
- **Relying on one LLM call for structure + retrieval hints** — Parser errors and shallow context; splitting **extraction** helped stability.

## How improvement was measured

- **Spot checks** on failure modes: off-topic refusal, compare with near-duplicate titles, single long “kitchen sink” hiring messages, high-volume paragraphs, and quota failures (fallback path vs hard error).

## Use of AI tooling

- **Cursor (agentic chat / edits)** — Used for implementation and refactors across `agent.py`, `retriever.py`, `catalog.py`, and prompts: slot gating, extraction robustness, compare resolution, dependency pins, and documentation. No separate no-code builder; the service remains a small hand-maintained FastAPI codebase.
