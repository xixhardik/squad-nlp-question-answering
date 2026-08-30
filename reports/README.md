# Reports

**This directory is version-controlled on purpose.** It is the project's evidence
trail.

Every training run writes a small, diffable record here:

```
reports/<run_id>/
├── metrics.json           EM, F1, validation loss, runtime, throughput,
│                          peak GPU memory, latency, example and feature counts
├── config.resolved.yaml   the fully merged configuration that produced the run
└── env.json               python/torch/transformers/datasets versions, CUDA
                           version, GPU name, git commit SHA + dirty flag
```

`run_id` is `<experiment>-<model>-<config_hash>-<UTC timestamp>`, so a result can
always be traced back to the exact configuration and code that produced it.

## Rules

1. **Metrics are never typed by hand.** The comparison table in the root
   `README.md` is generated from the `metrics.json` files in this directory. A CI
   check fails if the generated block is stale relative to these records.
2. **Every metric carries its denominator.** `total_examples` is stored alongside
   each score, so a number can never be quoted without the sample size it came
   from.
3. **Failed runs are recorded too.** An experiment that crashed or was skipped for
   budget reasons gets a record saying so. A documented omission is a legitimate
   result; a silent gap is not.
4. **Checkpoints do not live here.** Weights go to the Lightning model registry
   and, optionally, the Hugging Face Hub. See
   [`../docs/lightning-workflow.md`](../docs/lightning-workflow.md).

## Current contents

Empty. No training run has been executed yet, so there are no results to record.
