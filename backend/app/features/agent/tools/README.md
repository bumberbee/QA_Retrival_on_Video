# Agent tools (A4)

Create one `ToolContext` per question and call `default_registry().run(name, args, ctx)`.
Every registered tool returns `ToolResult(ok, data, error, latency_ms)`; input and
dependency failures are returned as errors. `calc.py` retains its original API.
No HTTP routes or state machine are installed by this toolset.

| Tool | Arguments (JSON) |
| --- | --- |
| `search_hybrid` | `q`, `top_k=20`, `ocr_weight=1.0`, `speech_weight=1.0` |
| `search_ocr` | `q` |
| `search_speech` | `q` |
| `rerank` | `q`, `chunks` |
| `get_slide` | `slide_id` |
| `get_window` | `video_id`, `t_start`, `t_end` |
| `neighbors` | `slide_id`, `k=1` |
| `calc` | `expr` |
| `verify_claims` | `claims` (string or list of strings), `evidence_map` |

The named Python wrappers accept the same arguments and keyword-only `ctx`.
The existing `calc(expr)` function remains unchanged; use its registered wrapper
to obtain a `ToolResult` and enforce the question budget.

Source tools return `data.evidence` with stable `id` and original `chunk_id`,
`segment_id`, `source_type`, `text`, `start_sec` and `end_sec` fields. They also
return `data.found`, ready for `features.search.retrieval.build_context_with_ids`.
Alternatively combine evidence lists with `base.evidence_to_found`. Request-local
E1/E2 IDs belong to the context builder; source IDs never depend on tool call order.

Search always delegates to existing `retrieval.search`; OCR/speech wrappers pass
its `sources` parameter. `top_k` caps returned evidence, within the existing
retrieval candidate/slide limits. Source weights multiply the retrieved parent
slide's fused score, then order the returned evidence; they do not change E5,
BM25 or RRF. A zero weight excludes that modality's evidence. `searcher` is an
optional fake/injection boundary; `encoder` is passed to the existing search.
Search owns its index loading; `index_loader(video_id)` supplies slide/window
reads and resolves minimal rerank candidates. No global retrieval state is patched.

Reranking imports the shared A3 interface inside the handler and passes `scorer`
through. A missing module or `RerankerUnavailable` produces a failed `ToolResult`.
Minimal candidates with only `chunk_id`/`text` are accepted. When they lack source
metadata, an injected index loader can resolve it; unknown candidates remain in
`data.chunks` but are omitted from citeable evidence. No timestamps are fabricated.

Slides use full cleaned OCR (raw OCR/chunks as fallbacks), not just incremental
indexed OCR. Neighbors exclude the anchor and ads and are ordered chronologically.
Speech windows select spans overlapping `[t_start, t_end)`, prefer aligned speech
spans, and preserve original source timing, including overlap at window edges.
`verify_claims` only splits sentences, records citations and marks every claim
`unverified`, with the explicit A6 NLI TODO; it never calls a model.

`agent_limits.yaml` configures the hard per-question limits; the equivalent JSON
is used when YAML is unavailable. Defaults: 6 tool calls, 2 search invocations
(rounds), 2 LLM calls, 240 seconds. Validated executions, including dependency
failures, consume calls. Invalid inputs, duplicate calls and budget refusals do
not. Defaults are materialized before duplicate fingerprinting; dictionary key
order and equivalent integral/float JSON numbers are irrelevant. Future S5/S6
code must call `ctx.budget.reserve_llm_call()`
before each gateway call, including answer regeneration.

Tool execution returns at the wall deadline and refuses later calls. In-flight
Python/native model operations cannot be forcibly stopped; a daemon worker may
finish after timeout and its result is discarded. Dependencies should provide
their own cancellation/timeout support if resource cancellation is required.

Tests use synthetic in-memory `LectureIndex` objects, fake searchers/encoders and
a fake shared reranker module. They do not read real indexes or load models.
