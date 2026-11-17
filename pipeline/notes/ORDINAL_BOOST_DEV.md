# Ordinal boost: dev-120 before and after

Measured 2026-11-17 on the local CPU with the pinned E5 ONNX encoder, after the ordinal boost
("nhận định 14", "câu 2", "bài 3") was merged. Both runs use the same indexes
(`D:/thesis-data/run-20261102/ablation/indexes/recommended-final`), questions and options; the
only difference is `LECTURE_ORDINAL_BOOST`.

```
python pipeline/eval/eval_retrieval.py --index-root <indexes> --name dev120-ordinal-off --output-root D:/thesis-data/runs-exp   # LECTURE_ORDINAL_BOOST=off
python pipeline/eval/eval_retrieval.py --index-root <indexes> --name dev120-ordinal-on  --output-root D:/thesis-data/runs-exp   # default (on)
```

| Boost | Hit@1 | Hit@3 | MRR | Context recall |
|---|---:|---:|---:|---:|
| off | 0.636 | 0.818 | 0.718 | 0.809 |
| on | 0.636 | 0.827 | 0.723 | 0.818 |

The "off" run reproduces the previous evaluation (`verify-final`) exactly.

Two of 120 questions changed their slides or context:

- `tHE1mC_n0-E_q05` ("Câu 18 cho đơn vị SI của nhiệt nóng chảy riêng là gì?"): no hit before,
  now at rank 2 with its key terms in the context.
- `L25_V008_q10` ("... ở câu 3 là gì?"): already a rank-1 hit; the context changed, the metrics
  did not.

No question regressed. The dev set was written by AI and has been used for tuning, so this is
a development check, and one question is all the gain rests on.
