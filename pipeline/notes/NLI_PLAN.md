# A6 multilingual sentence support: pending real-model evaluation

No real model was loaded, no model measurement has been run, and no accuracy or
macro-F1 result for multilingual lecture claims is available. All automated
checks for this implementation use synthetic scorers. Thresholds are provisional:
Q10 accept = 0.6 and reject_contradict = 0.5. TODO A8: calibrate both on dev, then
freeze them before testing. Q10 reject = 0.2 remains the generic decision-zone
boundary and is not the contradiction threshold.

## Obtain and export separately

Candidate family: mDeBERTa-v3 base fine-tuned on XNLI / multilingual NLI, with a
three-way entailment, neutral, contradiction classifier. The exact repository ID
and revision hash could not be verified offline.
TODO: review the checkpoint's model card, Vietnamese coverage, license, tokenizer,
label order, and ONNX export compatibility; record the exact repository ID and
immutable revision hash here before provisioning. Do not assume a repository name
or resolve an unreviewed `main` revision.

In a separate authorized environment, obtain the reviewed, pinned checkpoint and
export a sequence-classification ONNX model using the export toolchain appropriate
to that architecture (for example Optimum with Transformers and PyTorch there).
A typical command, to be validated for the chosen checkpoint, is:

```text
optimum-cli export onnx --model /local/pinned-checkpoint --task text-classification /local/nli-export
```

Save a fast-tokenizer `tokenizer.json` (including its pair postprocessor) and the
checkpoint's `config.json` alongside the export. Keep any external ONNX tensor
data beside its model. Check dynamic batch and sequence axes, CPU ONNX Runtime
compatibility, and numerical agreement with the original checkpoint on a small
audited Vietnamese/English sample before transferring the files. This export
recipe has not been executed or validated here. Backend serving uses only
onnxruntime, tokenizers and NumPy; it never imports torch/transformers, downloads,
or resolves a Hub snapshot. Store assets outside the repository.

## Local directory and serving

```text
/local/nli-export/
  model.onnx                 # alternatively onnx/model.onnx
  tokenizer.json
  config.json
  [external ONNX data files beside model.onnx, if the export uses them]
```

`config.json` must contain semantic `id2label` names, for example:

```json
{"id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"}}
```

This is an example, not a fixed order: the scorer reads the checkpoint's mapping.
Generic `LABEL_0` names are rejected rather than guessed. Preserve the reviewed
checkpoint's actual mapping. Inputs are `input_ids`, optionally `attention_mask`
and `token_type_ids`; output is an N-by-3 logits array. Each pair is at most 512
tokens including specials; premise tokens are removed first, retaining the
hypothesis unless it alone exceeds the budget. Batches contain at most eight pairs.

Set `LECTURE_NLI_MODEL_DIR` to the provisioned directory. Set `LECTURE_NLI=on`
before starting the backend (default off; read once at import/startup). S6 applies
when the existing `LECTURE_AGENT=on` path generates an answer. Missing assets or
inference errors become `nli_error` in verification, preserving the answer and
the preexisting checks. The verify_claims tool keeps its historical NLI_TODO
placeholder if NLI is disabled or unavailable. No server was started for A6.

## Future labels and evaluation

Use UTF-8 CSV with columns `claim,evidence,label`, one sentence/premise pair per
row. `label` must be exactly `supported`, `unsupported`, or `contradicted`. Quote
fields containing commas or newlines with standard CSV quoting. Example authored
format only (not a measured dataset):

```csv
claim,evidence,label
"The lecture defines conditional probability.","The lecture defines conditional probability.",supported
"The lecture covers astronomy.","The lecture introduces conditional probability.",unsupported
"The event is impossible.","The event is certain.",contradicted
```

Have lecturers review multilingual labels and separate dev/test pairs by lecture
to avoid leakage. Sentence citations and inference-only sources should be audited
separately; pair classification alone does not measure end-to-end citation recall.

Later, with local assets and the backend inference dependencies available:

```text
python pipeline/eval/nli_eval.py /local/dev.csv --output /local/nli-dev.json
python pipeline/eval/nli_eval.py /local/synthetic.csv --scorer-fake
```

The CLI uses the production `nli.score_pairs`, prints accuracy, macro-F1 over all
three labels, and a confusion matrix (rows = gold, columns = predictions, order
supported/unsupported/contradicted). It sweeps accept/contradict thresholds using
one scoring pass; CLI flags can override the sweep. `--scorer-fake` uses only
exact text equality, the synthetic prefix `FAKE_CONTRADICT:`, and neutral results
otherwise; it never inspects gold labels and must not be reported as model quality.
No real-model or dev/test measurement has been performed.
