# Agent (A5)

`graph.run(question, video_id=..., ...)` walks a fixed path; code decides every step and no
model is called inside it.

```
S1 understand  route_question (Q4 cascade: rules, then E5 examples when an encoder is given)
S2 plan        template reply | question back | refuse | search
S3 search      search_hybrid through the tool registry (budgeted)
S4 grade       Q8: reranker score when present, otherwise E5 cosine
               enough -> S5 | not enough -> S3 once more with the typed question | irrelevant -> S7
S5 answer      handed back to features/ask (generation, citations, repairs)
S6 check       verify.py: arithmetic of the answer; one regeneration with a hint when it is wrong
S7 refuse      fixed refusal + the closest part of the lecture
```

Enable with `LECTURE_AGENT=on`. It applies to questions on one video; several ticked videos,
and any error inside the agent, use the baseline path. The API response and the stream's
`done` event then carry `refusal` and `agent` (states, Q4 decision, tool calls, Q8 per round,
S6 reports, the verification verdict, budget). The same trace is written to the `qa_answer` log line with the question, which is
the raw material for the mini-Jev dataset (A7b).

What it changes compared with the baseline:

- Greetings and thanks get a template reply; no search, no LLM.
- A bare pointer ("cái này là gì?") with no playing position asks which slide is meant.
- A question the rules mark out of scope is refused before searching.
- Evidence graded `irrelevant` is refused without calling the LLM, with a pointer to the
  closest slide. Any other "out of scope" decision is refused unless the evidence is `enough`.
- A weak first search is retried once with the question as typed, when that differs from the
  rewritten query.
- A written equality that does not hold ("C(12,4)/C(15,4) = 99/667") triggers one regeneration
  whose prompt states the exact value. The better of the two answers is kept. When streaming,
  the first answer is already on screen and the corrected one replaces it through `done.answer`.
- Every generated answer gets a verdict, shown under it in the chat: `verified`, or `unverified`
  with reasons (`weak_evidence`, `numeric_mismatch`, `invalid_citation`, `no_citation`,
  `inferred_citation`). "Verified" means these checks passed; whether each sentence follows
  from its source (Q10, NLI) is not checked.

Deliberately not here yet: per-type plans beyond the shared search (`search_ocr`, `neighbors`,
`get_window`, splitting comparisons), LLM query rewriting for the retry, sentence-level support
(A6) and the Gateway guard (S0). The per-type plans need the test set to show they help.

Limits of what is measured: on the 120 dev questions with a fake search, no question got a
template reply, a question back or a pre-search refusal. Rules alone decided 62 of them;
the rest fell to the default type, and a digit in the question is enough for `formula`.
The Q8 reject threshold comes from dev-120 (`pipeline/notes/Q8_SIMILARITY.md`); nothing here has
run with the real encoder or LLM.
