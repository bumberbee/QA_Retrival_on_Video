# Decisions (A2 / A2b)

This slice is independent of the HTTP API. Import `decide`, `Decision`,
`route_question`, `QuestionDecision`, and `resolve_route` from
`app.features.decisions`. `BACKENDS` and `POLICY` are also exported.

```python
decide("Q3", {"question": "Còn cái đó thì sao?"})
decision = route_question("Công thức tính phương sai?", {})
resolve_route(decision)  # ("run", "plan_formula")
```

## Policy and probabilities

`agent_policy.yaml` is loaded using installed PyYAML. Without PyYAML,
`agent_policy.json` supplies the same policy. The example bank has the same
YAML/JSON fallback. Neither dependencies nor models are installed or downloaded.

Binary `p` is the probability of the first policy label. A confident negative
decision therefore has low `p`, high `abs(2*p - 1)` confidence, and a reject zone.
Choice `p` is the selected label's probability; confidence is
`max(0, (p - 1/n)/(1 - 1/n))`. Semantic cosine scores follow the same interface
but are not calibrated probabilities. Zones use `p >= accept`, `p <= reject`,
otherwise uncertain. Q2's accept zone means accepting its selected classification;
callers must inspect its label to determine whether a question is safe.

`decide(name, state, *, backend=None, policy=None, backends=None)` permits isolated
dependency injection. Backends implement `run(name, state)` returning
`BackendResult(label, p, scores={})` or a matching dict. Only `NotConfigured`
triggers fallback: configured backend -> rules -> neutral prior with
`backend="fallback"` and confidence zero. Q6/Q8/Q9/Q10/Q11 have no invented
rules; callers must handle their uncertain priors. Unknown names/backends and
invalid outputs raise errors. Latency includes failed attempts and fallback.

## Reranker scores (A3)

Set `LECTURE_RERANK=on` before starting the backend; the default is off. Search
reranks the top 30 RRF candidates before the existing slide selection. With the
flag off, result/context/evidence-ID bytes match the pre-A3 fixture. With scores,
search adds `best_rerank` and scored evidence chunks retain `rerank_score` and
one-based `rerank_rank`, including in the context returned by `build_context`.
Unscored same-slide/neighbor expansions have no invented scores.

```python
found = search(video_id, question)
decide("Q8", found)  # or {"retrieval": found}
decide("Q9", {"chunk": evidence_chunk})  # or the scored chunk itself
```

The registered `reranker` backend consumes existing scores without loading a
model. Q8 uses `best_rerank`: `>= accept` -> `enough`, `<= reject` -> `irrelevant`,
otherwise `insufficient`. Q9 uses the individual chunk's `rerank_score`:
`>= accept` -> `keep`, otherwise `drop`. The Q9 keep threshold is its policy
`accept` value. Injected policies control labels and zones as well as defaults.
Missing scores raise `NotConfigured` and retain the existing neutral fallback;
callers must handle this uncertain result for unscored expansions.

For Q8 specifically, `p` is the evidence support score, so the returned zone
matches the accept/reject thresholds; it is **not** a calibrated probability
of the selected three-way label. The generic choice confidence formula is
still reported, but does not establish calibration. Q9 `p` is support for `keep`.

`search/rerank.py` implements the shared `rerank(question, candidates, *,
scorer=None, top_k=None)` interface. Injected scorers return logits. The default
loads BAAI/bge-reranker-v2-m3 lazily, preferring installed ONNX/tokenizers, with
joint 512-token truncation and batches of eight. It only reads local files:
set `LECTURE_RERANK_MODEL_DIR` to an already provisioned snapshot/export, or
fill the `RERANK_REVISION` commit-hash TODO to resolve an existing Hub cache.
No `main` fallback or download is attempted. Missing files/dependencies raise
`RerankerUnavailable`; search propagates that error when enabled.

See `pipeline/notes/RERANK_PLAN.md` for the pending paired dev-120 measurement
and Kaggle notebook builder recipe. No evaluation/model run was performed for A3.

## Router injection and state

`route_question(question, state, *, encode=None, llm=None, backends=None,
policy=None)` stops after the first passing cascade step. Its decision contains
`steps`, `step_timings_ms` (including unavailable steps), and `decided_by`.
It copies state and does not modify the session.

- `previous_question`, `previous_qtype`, or `previous_decision` (object/dict) give
  context for follow-up inheritance and pointer clarification.
- `current_time` may be zero. With no timestamp and no prior context, a bare
  pointer asks which slide/minute the user means.
- `qtype` or `question_decision` supplies Q4's result for `decide("Q5", state)`.
- `lecture_terms` contains terms from the selected lecture. `top1_rrf` is a raw
  RRF score; missing scores do not justify an out-of-scope decision.
- `scope` defaults to `"single_video"`; use `"multi_video"` for wider scope.
  `mentioned_video_ids` identifies videos referenced in the question. Explicit
  phrases such as "hai video" / "both videos" also trigger single-scope clarify.
- `answer` and/or `claims` provide text for Q12's arithmetic, equality, percent,
  fraction and combinatorial-expression detection. Dates/citation IDs alone do
  not demand calculation.
- Inject `encode(text) -> vector`, `state["encode"]`, or an already-created
  `state["encoder"]` exposing `.encode`. Existing search `OnnxEncoder.encode`
  fits directly. Hashable encoders reuse cached example embeddings. Alternatively
  inject a persistent `SemanticBackend(encode=encoder.encode)` in `backends`.
- `llm(question, state)` (or `state["llm"]`) returns `{qtype, modality}` as JSON
  or a mapping, optionally with `p`. Invalid JSON/types select `explanation`;
  invalid modality derives from Q4. No LLM configured also returns the neutral
  explanation default. Code never reads a proposed route/action from the LLM.
- Replace entries of `BACKENDS`, or inject `backends` overrides into the router,
  to configure mini-Jev/Jev. Default attempts are mini-Jev then Jev; override
  order with `q4_backends`, or select one with `q4_backend` / policy Q4 backend.
  These adapters are currently stubs; no remote calls are made.

Two close types (< 0.05) ask for clarification only when all informative scoring
steps remain tied and lead to different routes. A later clear step resolves an
earlier tie. `resolve_route` returns the exact A2b route table, or
`("clarify", Vietnamese_prompt)` for clarify actions; unknown types raise KeyError.

## Remaining work

- A8: calibrate decision/router/RRF thresholds and cosine probabilities on dev;
  record accuracy, macro-F1, and stopping rates; freeze before evaluating test.
- Review the 20 authored examples per label with lecturers; no evaluation data
  was used to author them.
- Agree Q6/Q11 meanings and labels: A2/A2b do not define them.
- Provision/pin the reranker weights; configure NLI, mini-Jev/Jev and an LLM callable when available.
- The agent state machine (A5 part 1) is in `features/agent/graph.py`; frontend debug display is deferred.

## Interim Q8 without a reranker

`decide("Q8", found, backend="similarity")` grades evidence by the search's best E5 cosine
(`best_semantic`), with thresholds under `Q8.similarity` in the policy (accept 0.85, reject 0.84).
The reject threshold was chosen on dev-120 (`pipeline/notes/Q8_SIMILARITY.md`); the signal is weak
and the reranker should replace it. `p` is the cosine, so the policy's
reranker zones do not apply to it: read the label. The agent uses the reranker backend when the
search carries `best_rerank` and this one otherwise.
