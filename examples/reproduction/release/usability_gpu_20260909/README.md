# Milena exit-mode validation — September 9, 2026

The [summary](summary.json) binds the source commit, runner, protocol manifest
and runtime image. Both runs used one idle NVIDIA A100-SXM4-80GB, batch size 1
and the pinned public EVA 1.4B CLM checkpoint. All 135 fresh predictions in each
run exactly equaled the previously saved new-inference vector in
[expected/predictions.csv](../../milena_14b/expected/predictions.csv).

- [Default report](default_report.json): complete inference, metric calculation
  and plots; exit code **0**.
- [Strict report](strict_report.json): the same successful execution with
  `--strict-reference`; exit code **2** for the retained reference difference.

Spearman remains 0.8394237924835843 versus 0.8360456283218484. These records
validate the new execution/comparison distinction; they do not claim that
all paper experiments have been reproduced. Reports retain the original
comparison fields. Full-process elapsed time in the summary also includes
Python startup; `inference_seconds` in each report measures scoring only.
