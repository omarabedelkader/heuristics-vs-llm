# Package split validation

Validated on 2026-10-06.

- Pharo regression suite: **94 tests passed** in the Pharo 14 validation image.
- Python regression suite: **9 tests passed**, including overlap, missing-package,
  duplicate-package, and contaminated-row rejection.
- Loaded the updated baseline through Metacello into a clean cached Pharo 14
  image. Fresh pipeline setup selected **50 benchmark / 679 training packages**.
- Ran both actual shell pipelines against a disposable image with three tiny
  fixture packages: one held out, two used for training. Normal LLM calls used
  the repository's test transport; re-ranking used actual PyTorch training,
  ONNX export/parity checks, and the HTTP scoring service.
- Verified that normal and neural tables were exported, the split was identical
  across both pipelines and model metadata, training/test rows were disjoint,
  and the service/experiment lock were cleaned up.
- Verified that changing the count or seed of an existing experiment fails
  before benchmarking and leaves the saved split intact.
- Deliberately appended a held-out benchmark row to training data: the trainer
  rejected it before creating a model.
- Shell syntax and Git whitespace checks passed in both repositories.

No full-image training or 50-package LLM benchmark was run during validation.
The smoke results establish workflow correctness, not model quality.

An additional older cached Pharo 14 snapshot (`cf2bfb5`) reported 27 compiler
errors in existing test fixtures (including unreachable statements). Running
the unmodified repository tests on that same image reproduced all 27 errors.
The new split tests passed there; the complete suite passed in the validation
image (`359c9be`). Metacello loading and the 50-package setup check also passed
in the older snapshot.

Detailed local smoke artifacts are in
`/private/tmp/coo-package-split-work/pipeline-smoke/`. The independent fresh
50-package setup is in `/private/tmp/coo-package-split-work/setup-smoke/`.
