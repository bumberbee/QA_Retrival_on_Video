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
- Configure reranker, NLI, mini-Jev/Jev and an LLM callable when available.
- API, agent state machine and frontend debug integration are intentionally deferred.
