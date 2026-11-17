# Agent (A5 part 1)

`graph.run(question, video_id=..., ...)` walks a fixed path; code decides every step and no
model is called inside it.

```
S1 understand  route_question (Q4 cascade: rules, then E5 examples when an encoder is given)
S2 plan        template reply | question back | refuse | search
S3 search      search_hybrid through the tool registry (budgeted)
S4 grade       Q8: reranker score when present, otherwise E5 cosine
               enough -> S5 | not enough -> S3 once more with the typed question | irrelevant -> S7
S5 answer      handed back to features/ask (generation, citations, repairs)
S7 refuse      fixed refusal + the closest part of the lecture
```

Enable with `LECTURE_AGENT=on`. It applies to questions on one video; several ticked videos,
and any error inside the agent, use the baseline path. The API response and the stream's
`done` event then carry `refusal` and `agent` (states, Q4 decision, tool calls, Q8 per round,
budget). The same trace is written to the `qa_answer` log line with the question, which is
the raw material for the mini-Jev dataset (A7b).

What it changes compared with the baseline:

- Greetings and thanks get a template reply; no search, no LLM.
- A bare pointer ("cái này là gì?") with no playing position asks which slide is meant.
- A question the rules mark out of scope is refused before searching.
- Evidence graded `irrelevant` is refused without calling the LLM, with a pointer to the
  closest slide. Any other "out of scope" decision is refused unless the evidence is `enough`.
- A weak first search is retried once with the question as typed, when that differs from the
  rewritten query.

Deliberately not here yet (part 2): per-type plans beyond the shared search (`search_ocr`,
`neighbors`, `get_window`, splitting comparisons), LLM query rewriting for the retry,
`calc`/citation checks and one regeneration (S6), and the Gateway guard (S0).

Limits of what is measured: on the 120 dev questions with a fake search, no question got a
template reply, a question back or a pre-search refusal. Rules alone decided 62 of them;
the rest fell to the default type, and a digit in the question is enough for `formula`.
Thresholds are not calibrated, and nothing here has run with the real encoder or LLM.
