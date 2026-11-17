# Q8 by E5 cosine: threshold chosen on dev-120

The agent's interim evidence gate refuses a question when the best E5 cosine of its search
(`best_semantic`) is at or below `Q8.similarity.reject`. This note records how that number
was chosen. It replaces the earlier guess of 0.80.

## Data

`D:/thesis-data/runs-exp/verify-final/retrieval.json` (measured 2026-11-02, SHA-256 starting
`55ecb7784a857a77`): the 120 dev questions, 110 answerable and 10 unanswerable, with the
`best_semantic` of each. No model was loaded and no search was rerun for this note. The ordinal
boost merged later changes ranking, not cosines, so these scores still apply.

The dev questions were written by AI and have been used for tuning. Ten unanswerable questions
is a very small sample. Treat the numbers below as a development choice, not a result.

## Table

```
python pipeline/eval/calibrate_q8.py D:/thesis-data/runs-exp/verify-final/retrieval.json
```

| Reject at or below | Answerable refused | of which retrieval had Hit@3 | Unanswerable caught |
|---|---|---|---|
| 0.800 | 2/110 (1.8%) | 1 | 0/10 |
| 0.820 | 3/110 (2.7%) | 2 | 1/10 |
| 0.830 | 7/110 (6.4%) | 4 | 2/10 |
| 0.835 | 8/110 (7.3%) | 5 | 4/10 |
| **0.840** | **9/110 (8.2%)** | **5** | **6/10** |
| 0.845 | 13/110 (11.8%) | 7 | 7/10 |
| 0.850 | 17/110 (15.5%) | 11 | 7/10 |
| 0.860 | 27/110 (24.5%) | 20 | 9/10 |

Answerable questions range from 0.746 to 0.933 (median 0.874); unanswerable ones from 0.817
to 0.871. The two groups overlap heavily.

## Choice

Rule from the plan (A8): the highest threshold that wrongly refuses at most 10% of answerable
questions. That is **0.84**. The earlier 0.80 caught none of the unanswerable questions.

`accept` stays at 0.85 and is not calibrated. It only decides the "verified" flag, the retry
and the check of model-made "out of scope" decisions.

## What this shows

E5 cosine is a weak signal for "is there enough evidence": at the chosen threshold it still
lets 4 of 10 unanswerable questions through and refuses about 1 in 12 answerable ones, five of
which had the right slide retrieved. The reranker score is the intended signal for Q8; this
gate is the fallback until the reranker is measured (`RERANK_PLAN.md`). The thresholds must be
frozen before the held-out test set is run and must not be tuned on it.
