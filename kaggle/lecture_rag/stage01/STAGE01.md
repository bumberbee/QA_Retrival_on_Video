# Stage 01: measured extraction fixes

2026-11-02 · branch `upgrade/extraction-fixes` · no commit, push, or branch change.

The changes in `notebook_code/detection.py` are opt-in. The default detector reproduces **the complete archived intervals.json exactly on all ten extra videos**, including timestamps, keyframe hashes, reveal membership, and sharpness scores. The recommended slide configuration preserves the thesis's **L25 held-out F1 0.970** and **LPM F1 0.909 at ±1 s / 0.935 at ±3 s**. Board and editor options are scoped to the audited videos; their content-state precision has not been independently labelled.

## Method and compatibility

* **Board content:** an additional 384×216 content ROI, 6×4 local cells, signed foreground strokes, and directional additions/deletions. Five coarse samples form a temporal median, excluding a sample-local green-board presenter mask. Large non-green components cover clothing; warm saturated pixels cover skin. This is a CPU colour heuristic for this green board and white chalk, not a general person model. Hidden regions retain their last visible history. Masked holes use the board's background level rather than black, preventing artificial foreground edges. Two checks confirm a change; minimum state duration is 15 s. An early pending change is clamped to the duration constraint rather than left permanently ineligible.
* **Snapshots:** board intervals receive explicit periodic fallback states at 120 s, with balanced tails that avoid tiny intervals. Original refined boundaries remain intact. A `board_audit.json` distinguishes `local_board_change` and `periodic_content_snapshot`. The final board run discovered enough local changes that its maximum state was only 63.92 s.
* **Replacement safety:** `reveal_thresholds={'replacement_safe': True}` checks signed stroke survival without subtracting the persistent editor/template regions. A 16×9 grid vetoes locally deleted/replaced content even when global pixel growth rises. It also checks the group anchor and directional OCR word retention when optional OCR is supplied. Ordinary slide reveals retain the legacy predicate by default.
* **Content identity:** `merge_same_content=True` extracts a conservative bright rectangular screen, removes its bezel, and resizes it before comparing. Both directions require ≥0.96 stroke retention, enough content, no deleted regions, and limited image change. A verified layout switch promotes the equivalent content in the new view to the spatial anchor; subsequent reveals therefore do not fail against the old PiP layout. Same-layout merges do not repeatedly reset the anchor. No page identity is inferred through a lecturer-only shot, and perspective correction is not implemented.
* **Keyframes:** optional `keyframe_options` scores ROI sharpness, visible persistent content, presenter occlusion, and temporal deviations. Boards prioritize exposed writing and can keep a second actual frame when it reveals ≥2% additional persistent content. Candidates and complementary frames stay inside the state's half-open interval. The ordinary whole-frame sharpness selector is unchanged when the option is absent.

`export_state_onsets=True` adds, without renaming existing fields:

| Location | Added fields |
|---|---|
| Each reveal state | `content_onset_sec`, `content_observed_sec`, `content_onset_bounds_sec`, `onset_method`, `content_delta` |
| Each grouped interval | `representative_content_onset_sec`, `visual_evidence_windows`, `long_dynamic_review` |
| Optional observations | `content_quality`, `complementary_keyframes`, `display_roi` |

`content_delta` reports pixel counts and normalized changed/deleted regions, not recognized text lines. State onset is a detector boundary; the observed timestamp identifies the actual saved frame. The bounds explicitly preserve uncertainty for writing inside a state. **Per-line OCR onset is not invented.** Group start remains unchanged for backward compatibility; downstream visual citations must consume the state windows/observed timestamps.

## Ten-video comparison

Inputs were read-only: `D:/thesis-data/mp4-extra` and `D:/thesis-data/run-20261020/export`. All experiment outputs are under `runs/stage01`. Comparisons keep the archived thresholds, sampling, hidden-change splitting, and motion masking. Ad scorer, slide classifier, and merge OCR remain disabled, as in the audited run.

| Video | Intervals before → after | States before → after | <5 s before → after | >120 s before → after | Longest before → after, s |
|---|---:|---:|---:|---:|---:|
| cs_0001 | 21 → 21 | 49 → 49 | 1 → 1 | 2 → 2 | 131.17 → 131.17 |
| D_kuulxKuxc | 103 → 101 | 150 → 150 | 20 → 19 | 0 → 0 | 75.60 → 76.28 |
| hVU5Mw19V_c | 18 → 117 | 26 → 117 | 2 → 6 | 4 → 0 | 650.40 → 63.92 |
| L25_V008 | 15 → 13 | 67 → 67 | 2 → 2 | 4 → 4 | 265.50 → 265.50 |
| L25_V011 | 10 → 10 | 23 → 23 | 2 → 2 | 3 → 3 | 282.76 → 282.76 |
| L25_V037 | 10 → 10 | 21 → 21 | 2 → 2 | 3 → 3 | 299.92 → 299.92 |
| tHE1mC_n0-E | 203 → 203 | 268 → 268 | 65 → 65 | 0 → 0 | 88.76 → 88.76 |
| v2YTz8ToEag | 43 → 43 | 72 → 72 | 5 → 5 | 8 → 8 | 372.49 → 372.49 |
| y1CvB950VlU | 54 → 70 | 81 → 81 | 16 → 25 | 0 → 0 | 62.83 → 67.03 |
| _eDGTHhyQ7M | 25 → 25 | 36 → 36 | 7 → 7 | 8 → 8 | 330.46 → 330.46 |
| **Total** | **502 → 613** | **793 → 884** | **122 → 134** | **32 → 28** | **650.40 → 372.49** |

More intervals are intentional on the dynamic board and editor, and increase OCR/indexing work. The short-interval count rises because original short board states are preserved and unsafe editor joins are rejected; it is not evidence of improved boundary precision. The 28 remaining long intervals have review flags. All ten outputs have full duration coverage, no overlap/gap, and valid state, primary-keyframe, and complementary-keyframe timestamps.

The board's formerly single **0–578.96 s** state is now **34 states**, maximum 63.92 s. The formerly single **775.36–1425.76 s** state is now **42 states**, maximum 27.08 s. The full video has **91 additional local splits** and **40 complementary keyframes**. These fine board states are evidence observations, not 117 independently validated slide identities.

Machine-readable measurements: [final-measurements.json](../runs/stage01/final-measurements.json). Full contact sheets for all ten videos: [evidence directory](../runs/stage01/evidence/).

## Audited failure cases

| Failure | Result and limit |
|---|---|
| Missing board writing and “Loại I/II/III” | **Visual evidence recovered.** Class totals, the table, intermediate formulas, and the later example have separate observations. The example is visible in the saved frame at **1239.92 s**, with subsequent calculations at 1297.48 and 1422.52 s. Board boundaries remain unlabelled; exact per-line onset still requires OCR. |
| y1CvB950VlU source states 58–60 merged despite replacement | **Grouping fixed.** Source states 58, 59, and 60 are separate groups; `split_deleted_or_replaced_content` prevents the old expression from being represented only by the replacement. Some selected frames still show active typing/tooltips. |
| cs_0001 late text cited from 34.7 s | **Export fixed; existing index not rebuilt.** Group start is 34.733333 s, while its representative state's explicit onset is **117.833333 s**, eliminating the previous 83.1 s ambiguity for downstream consumers. Every state has its own evidence window. |
| L25_V008 seg_00003/00004 same slide across views | **Fixed.** The definition slide becomes one **89.889967–130.396567 s** group. The later “Câu 1” view switch also merges, with ordinary reveals preserved by the new spatial anchor. Six labelled true slide changes remain detected. |
| D_kuulxKuxc view/content duplicates | **Two joins verified visually.** Three student views display the same “Mai có 13 tờ giấy màu…” question; they now share one content interval. The numbers/text do not change. |
| tHE1mC_n0-E alternating exam views | **Unresolved.** Still 203 intervals and 65 under 5 s. The conservative screen normalization does not bridge lecturer-only or unmatched/perspective views. |
| Lecturer occludes board keyframe | **Partially fixed.** ROI scoring and 40 complementary observations expose more writing; e.g. the Bayes formulas at 1040.44 s are much clearer than the old 1074.08 s representative. Some short writing states remain occluded. No synthesized board image is produced. |
| IDE popup at 641.7 s | **Unresolved for that state.** The 640.433333–642.9 s state contains the overlay throughout the samples, so the selected time remains 641.7 s. The following state has a clean frame at 646.833333 s, which is not substituted into the earlier interval. Persistent UI overlays need a UI mask or OCR exclusion. |

![Board evidence before and after](../runs/stage01/evidence/board_before_after.jpg)

![Slide view joins before and after](../runs/stage01/evidence/L25_V008_view_joins.jpg)

![Editor replacements and persistent overlay](../runs/stage01/evidence/code_before_after.jpg)

Additional verified join sheet: [D_kuulxKuxc](../runs/stage01/evidence/D_kuulxKuxc_view_joins.jpg).

## F1 regression: same thesis protocol

The 0.970 result comes from archived `gold/eval_threshold_sweep.py` → `eval_ad_detection.py`, located under `runs/legacy-gold`, and is described in `docs/make_progress_slides.py` and `docs/make_stage_figures.py`. The bundled [L25 labels](labels/l25_protocol.json) preserve its event metadata and ad edges. Four videos are tuning and six held-out; the protocol counts **S/SC** boundaries, uses **±1 s**, and treats **C/L** and nearby ad edges as neutral. Its duplicate-event handling is retained exactly. Some original label-source entries say “label to confirm”; this work reproduces that thesis gold set, not a new independent annotation campaign.

LPM uses the eight videos listed in `evaluate.py`, authors' slide starts, the annotated span only, and distance-sorted one-to-one matching at **±1 s and ±3 s**. The evaluator consumes `lpm-videos/slides.csv`; the source annotation dataset's `segments.txt` files and video mapping were also retrieved. All **352 slide starts match the original segments.txt sequences exactly on 8/8 videos**; see [label-source-check.json](../runs/stage01/label-source-check.json). The reported LPM score is separate from the L25 0.970 score.

| Set / tolerance | TP / FP / FN, before = after | Precision before = after | Recall before = after | F1 before → after |
|---|---:|---:|---:|---:|
| L25 tuning ±1 s | 110 / 2 / 4 | 0.982143 | 0.964912 | 0.973451 → 0.973451 |
| **L25 held-out ±1 s** | **97 / 1 / 5** | **0.989796** | **0.950980** | **0.970000 → 0.970000** |
| LPM ±1 s | 321 / 41 / 23 | 0.886740 | 0.933140 | 0.909348 → 0.909348 |
| LPM ±3 s | 330 / 32 / 14 | 0.911602 | 0.959302 | 0.934844 → 0.934844 |

Private CPU kernels, **0 GPU-hours used**:

* [Full scan and ablations](https://www.kaggle.com/code/thnhtrungnguynmif/stage01-regression-20261101): attached L25 and the 14 GB LPM dataset; froze baseline state timestamps.
* [Final anchor fix and parity evaluation](https://www.kaggle.com/code/thnhtrungnguynmif/stage01-regression-fixed-20261102): reconstructed the original JPEG-90 scan images with the same PyAV resize/PTS, then verified **exact original grouping on 18/18 videos** before applying the final identity logic. This avoids a lossy second-JPEG comparison and an unnecessary full rescan.

Final scores: [regression-fixed-scores.json](../runs/stage01/regression-fixed-scores.json); per-video parity/runtime: [runtime.json](../runs/stage01/regression-fixed/runtime.json). Original sweep scores were independently reproduced from both downloaded sweep kernels, including LPM 0.909348/0.934844.

**Rejected global setting:** strict replacement safety on all ordinary slides produces held-out L25 F1 **0.741818** and LPM ±1 s F1 **0.891034**. It rejects benign moving/shifted reveal content too often. Therefore it is not a new default and is recommended only for the audited editor. The slide identity option is the globally tested grouping change; board/editor options do not occur in the labelled conventional-slide set.

## CPU runtime and reproducibility

Local system Python **3.11.9**, PyAV **16.1.0**, OpenCV **5.0.0**, NumPy **2.4.6**, Pillow **12.3.0**, Windows, 16 logical CPUs. OpenCV used one thread; PyAV used automatic codec threading. `D:/FilmTools/Shotcut` supplied ffmpeg/ffprobe. No system package was installed and no extra venv was needed. Kaggle CPU kernels installed their media dependencies in their own environment.

| Video | Shared decode + baseline export/features, s | Additional after pass, s |
|---|---:|---:|
| cs_0001 | 65.0 | 6.9 |
| D_kuulxKuxc | 67.4 | 14.7 |
| hVU5Mw19V_c | 104.0 | 41.2 |
| L25_V008 | 78.4 | 10.7 |
| L25_V011 | 26.9 | 5.0 |
| L25_V037 | 32.0 | 9.6 |
| tHE1mC_n0-E | 273.1 | 86.9 |
| v2YTz8ToEag | 133.4 | 19.5 |
| y1CvB950VlU | 70.7 | 14.3 |
| _eDGTHhyQ7M | 122.1 | 16.2 |
| **Total, 279.4 video minutes** | **973.0** | **225.1** |

These are measured phases, not two independent end-to-end benchmarks. The shared scan exports baseline frames and extra content features; the after pass reuses exact coarse signatures and confirmed fine buffers. Their sum, **1198.1 s (~20 min)**, is a conservative reconstruction cost because some primary detection/export work is repeated. CPU/disk contention from concurrent work affected timings. The final anchor-only grouping refresh has separate `group_refresh_sec` values in the measurements and preserves state PTS/images. Failed disk-full attempts and exploratory sweeps are excluded from the table.

`board_sweep.py` compares local cell sensitivity using the exact coarse PTS schedule, omitting unused fine-frame pixel conversion. Final 8/12/18/24-pixel thresholds produced 111/111/111/117 board states; 24 remains the scoped default. The masked-hole and early-pending regression tests protect against motion artifacts and permanently suppressed changes. These counts are diagnostics, not a board F1 calibration.

From the repository root:

```powershell
$env:PATH = 'D:\FilmTools\Shotcut;' + $env:PATH
python kaggle/lecture_rag/stage01/experiment.py --purge-cache
python kaggle/lecture_rag/stage01/report_assets.py
python kaggle/lecture_rag/stage01/make_kernel.py
kaggle kernels push -p kaggle/lecture_rag/runs/stage01/kernel
python kaggle/lecture_rag/stage01/make_replay_kernel.py
kaggle kernels push -p kaggle/lecture_rag/runs/stage01/replay-kernel
python kaggle/lecture_rag/stage01/evaluate.py --result kaggle/lecture_rag/runs/stage01/regression-fixed/result --lpm-slides kaggle/lecture_rag/runs/stage01/labels-lpm-videos/slides.csv
python -m pytest kaggle/lecture_rag/tests -q
```

Use `kaggle kernels status` and `kaggle kernels output --file-pattern 'result\.zip$|runtime\.json$' --page-size 200`, then extract the result zip inside `runs/stage01`. `refresh_groups.py` is the optional state-preserving refresh after a grouping-only edit. Small LPM labels remain under `runs/stage01/labels*`; the large annotation payloads were removed. No LPM video was downloaded locally.

## Recommended builder configuration

The owner should add these arguments to `build_fast_pipeline.py`; it was not edited by this task. All function defaults remain legacy-compatible.

```python
stage01_options = {
    "export_state_onsets": True,
    "merge_same_content": True,
}
if video_id == "hVU5Mw19V_c":
    board_roi = (0.0, 0.08, 1.0, 0.96)
    stage01_options.update(
        board_options={
            "roi": board_roi, "window_samples": 5,
            "check_every_sec": 5.0, "min_state_sec": 15.0,
            "snapshot_sec": 120.0, "minimum_pixels": 24,
        },
        keyframe_options={"roi": board_roi, "mode": "board", "complementary": True},
        merge_reveals=False,
    )
elif video_id == "y1CvB950VlU":
    stage01_options.update(
        reveal_thresholds={
            "replacement_safe": True, "directional_threshold": 0.94,
            "deletion_minimum_pixels": 12, "deletion_fraction": 0.20,
        },
        keyframe_options={"roi": (0.26, 0.08, 0.99, 0.57)},
    )
# Retain sample_hz=1, fine_hz=8, change_threshold=.055,
# stability_threshold=.035, stable_samples=2, min_interval_sec=2.
```

The ROI must be inspected before using board/editor settings on another layout. Board detection and keyframe scoring require automatic single-pass scanning and the same ROI. The green/skin heuristic may mask coloured chalk or fail on other board colours; boards, angled screens, and live editors still need separate labelled validation.

## Tests and GPU OCR follow-up

The most recent required system-Python command returned **120 passed, 14 subtests passed, 2 failed in 59.57 s**; see [pytest-final.txt](../runs/stage01/pytest-final.txt). Both failures concern concurrent changes outside Stage 01: `test_embedded_sources_match_maintainable_python_sources` finds stale embedded `alignment.py` in the stage-04 notebook, and `test_harness_config_matches_generator_defaults` finds `school_signatures` in the builder versus `bounded_schools` in the text-phase evaluation snapshot. No Stage 01 test failed. This task did not edit notebooks, builders, alignment, or the text-phase configuration.

The new stage-01 tests cover additions, local replacement despite increased ink, directional OCR deletion, presenter masks, writing/erasure, hidden history, minimum-duration candidates, masked holes, periodic snapshots, view-switch anchors, complementary observations, persistent/transient selection limits, legacy schema defaults, onset export, and the exact evaluation protocols. All ten final output manifests passed signature and asset-checksum verification after the editor manifest refresh; every local report link resolves.

Stage 01 uses no GPU or OCR. **Old OCR/index exports remain read-only and still contain the old visual evidence/timestamps.** A GPU OCR rerun should read every required reveal state with its own evidence window, plus useful board complements, then rebuild stages 04/05. The new run has 884 source states plus 40 board complements (924 possible observations before hash deduplication). Prioritize hVU5Mw19V_c, y1CvB950VlU, and cs_0001; regrouped L25/D_kuul frames can reuse identical-image OCR where available. The existing final-state-only OCR path must not attach newly revealed lines to group start. Persistent IDE tooltip/autocomplete text also needs exclusion before indexing. ASR itself does not need a rerun solely for these stage-01 changes.
