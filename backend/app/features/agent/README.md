# Agent (A5)

`graph.run(question, video_id=..., ...)` walks a fixed path; code decides every step.
Generation remains in features/ask; S6 can additionally use a local NLI model.

```
S1 understand  route_question (Q4 cascade: rules, then E5 examples when an encoder is given)
S2 plan        template reply | question back | refuse | search
S3 search      search_hybrid through the tool registry (budgeted)
S4 grade       Q8: reranker score when present, otherwise E5 cosine
               enough -> S5 | not enough -> S3 once more with the typed question | irrelevant -> S7
S5 answer      handed back to features/ask (generation, citations, repairs)
S6 check       verify.py: arithmetic/citations and optional sentence support (Q10, A6)
               one shared regeneration: numeric mismatch first, otherwise contradiction
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
  `inferred_citation`). With `LECTURE_NLI=on`, sentence support also contributes
  `unsupported_claim` and `contradicted_claim`. "Verified" means the enabled checks passed;
  the NLI scores and thresholds have not been calibrated or measured with real weights.

Deliberately not here yet: per-type plans beyond the shared search (`search_ocr`, `neighbors`,
`get_window`, splitting comparisons), LLM query rewriting for the retry and the Gateway guard
(S0). The per-type plans need the test set to show they help.

Limits of what is measured: on the 120 dev questions with a fake search, no question got a
template reply, a question back or a pre-search refusal. Rules alone decided 62 of them;
the rest fell to the default type, and a digit in the question is enough for `formula`.
The Q8 reject threshold comes from dev-120 (`pipeline/notes/Q8_SIMILARITY.md`); nothing here has
run with the real encoder or LLM.

## Sentence support (A6 / Q10)

Set `LECTURE_NLI=on` before backend startup (default off, read once at import like
`LECTURE_RERANK`). Set `LECTURE_NLI_MODEL_DIR` to an already provisioned local ONNX
multilingual NLI export. Serving uses onnxruntime, tokenizers and NumPy only. No
torch/transformers, network calls, downloads, or repository model files are needed.
Loading is lazy on first use. The directory must contain `model.onnx` (or
`onnx/model.onnx`), `tokenizer.json`, and `config.json` with semantic `id2label`
names for entailment/neutral/contradiction in the checkpoint's actual order.
Pairs are capped at 512 tokens including specials, trimming the premise first,
and scored in batches of eight. Missing assets/libraries raise `NLIUnavailable`.

Each sentence is a hypothesis; each cited evidence text is a separate premise.
Evidence comes from `ask.postprocess.evidence_texts`, exactly as retained in the
prompt. An uncited sentence, or one with only inferred citations, uses all
available evidence texts. An explicit missing citation is not repaired with
unrelated evidence. S5 preserves raw E IDs internally as `found.nli_answer`
before rendering timestamps, so sources with identical timestamps remain distinct
in both buffered and streaming answers. Leading refusals stripped by the existing
repair are excluded from NLI. Answer sentences are never deleted or edited by NLI.

Support is the maximum entailment over selected premises. A claim is `supported`
when entailment is at least Q10 `accept` (0.6); otherwise it is `contradicted` when
the maximum contradiction probability reaches `reject_contradict` (0.5), and
`unsupported` otherwise. A claim without premise text is unsupported. Both
thresholds are marked TODO A8: calibrate on dev. Q10 `reject` (0.2) remains the
generic decision-zone boundary, separate from the contradiction threshold.

New verdict reason codes for future frontend mapping (no frontend changes in A6):

- `unsupported_claim`: one or more sentences did not reach the support threshold.
- `contradicted_claim`: one or more sentences reached the contradiction threshold.

Either reason makes the verdict `unverified`. A contradiction can request the
existing single regeneration with a Vietnamese hint naming the sentence. Numeric
mismatches keep priority and use the arithmetic hint. Unsupported claims alone
do not request regeneration. The retained answer minimizes numeric mismatches
first, then contradictions and unsupported claims; ties keep the first answer.
An NLI error cannot make a candidate look better or consume a regeneration.

Only additive verification fields are emitted, and only when NLI is enabled:

- `agent.verification.nli`: report for the retained answer; also present as
  `agent.s6[i].nli` for each checked generation.
- `nli.claims`: rows containing `claim_id` (C1, C2, ...), `text` (sentence with raw
  E citations), `status`, `entail`, `contradict`, `evidence_ids` (selected premises),
  and `premise_chars` (total original characters submitted for that claim).
- `nli.counts`: `total`, `supported`, `unsupported`, `contradicted`.
- `nli.accept` and `nli.reject_contradict`: thresholds used for the report.
- `agent.verification.nli_error` / `agent.s6[i].nli_error`: error type/message
  when checking failed. The answer and existing verdict checks remain available;
  this field is diagnostic, not an additional verdict reason code.

These fields also appear under `agent` in the stream's existing `done` event and
the existing `qa_answer` trace. No existing response fields change. With the flag
off, serialized reports, citation metadata, answer text and tool placeholders
retain their pre-A6 bytes; automated tests enforce this contract.

`verify_claims` returns these claim statuses when enabled/available. Otherwise it
retains its exact historical `unverified` placeholder and `NLI_TODO` text. Inject
`ToolContext(nli_scorer=...)` for tools, `verify.check(..., scorer=...)` for S6, or
`nli.score_pairs(..., scorer=...)` / `nli.check_claims(..., scorer=...)` directly.
Injected scorers accept `(premise, hypothesis)` pairs and return normalized
`(entailment, neutral, contradiction)` triples. Direct helpers and the offline
evaluator are independent of the serving flag.

See `pipeline/notes/NLI_PLAN.md` for pending checkpoint review/export, the CSV
label format and the evaluation CLI. No real NLI model or dataset measurement
has been run; fake-scorer metrics are only test fixtures.
