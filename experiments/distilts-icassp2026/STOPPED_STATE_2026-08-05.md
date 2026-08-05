# DistilTS experiment stop snapshot

- Snapshot time: 2026-08-05 08:20 Asia/Taipei
- Core jobs recorded complete: 14 / 30
- Failed jobs: 0
- Interrupted job: `Table2_OnlyFTA_ETTm2_p192`
- Last logged progress before stop preparation: epoch 2, iteration 700 / 1043
- The interrupted job completed epoch 1 and wrote a checkpoint.
- Queue runner before stop: PID 150987
- Training worker before stop: PID 658807
- Resource monitor before stop: PID 161720

The experiment was intentionally stopped at the user's request. Completed jobs,
raw logs, the manifest, resource telemetry, checkpoints, reproduction scripts,
reports, and local code changes are retained. The interrupted job is not counted
as a completed formal result because it did not reach final test metrics and
profiling. Restarting `reproduce_core.py` will skip accepted completed jobs and
rerun the interrupted job from its normal job entry point.
