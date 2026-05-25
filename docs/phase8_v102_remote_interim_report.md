# Phase 8 v1.0.2 Remote Interim Report

Date: 2026-05-24

This is the interim remote-run state for the current `softcpsrecsimulator==1.0.2`
and Algorithms image:

```text
image_tag = sha-c8fbe78
dataset   = citylearn_challenge_2022_phase_all_plus_evs
```

The purpose is to keep the long remote waves readable while runs are still
queued/running.

## Artefacts

Unified scorecard:

```text
runs/remote_results/phase8_v102_unified_scorecard_2026_05_24/phase8_v102_scorecard.csv
runs/remote_results/phase8_v102_unified_scorecard_2026_05_24/phase8_v102_scorecard.md
```

Builder:

```text
scripts/build_phase8_v102_remote_scorecard.py
```

Focused validation:

```text
.venv/bin/python -m pytest tests/test_build_phase8_v102_remote_scorecard.py -q
3 passed
```

## Current Remote Counts

Latest unified scorecard snapshot:

```text
status:
  finished = 20
  failed   = 3
  running  = 2
  queued   = 7

waves:
  wave1_profile              = 18
  wave2_baselines            = 6
  wave2_retry8h              = 2
  wave3_maddpg_direct        = 3
  wave4_maddpg_v48_teacher   = 3
```

## Completed Baseline Readout

Finished Wave 2 baseline rows:

| Policy | Cost EUR | EV min feasible | EV within tolerance | Grid violation kWh | Runtime |
|---|---:|---:|---:|---:|---:|
| RandomPolicy | 10938.64 | 0.1568 | 0.0703 | 0.0000 | 0.285 h |
| NormalNoBatteryPolicy | 21625.66 | 1.0000 | 0.0523 | 0.0000 | 0.257 h |
| NormalPolicy | 22202.69 | 1.0000 | 0.0523 | 0.0000 | 0.278 h |

Current reading:

```text
RandomPolicy is only a sanity/lower-bound row. The cost is low because it fails
EV service badly.

NormalNoBatteryPolicy and NormalPolicy are valid operational references because
they satisfy EV minimum service and have zero grid violations.

NormalPolicy is worse than NormalNoBatteryPolicy on this snapshot, which means
the simple battery behaviour is not automatically useful under this scorecard.
```

## Timeouts And Retries

Initial Wave 2 CPU RBC rows that timed out with `02:00:00`:

| Policy | Original job | Slurm state | Runtime |
|---|---|---|---:|
| RBCBasicPolicy | f2a00ccf-5eb5-47df-ba90-65c886ffe847 | TIMEOUT | 2.008 h |
| RBCSmartPolicy | e7b24278-4bc2-4d32-b878-fcc679f70456 | TIMEOUT | 2.013 h |

Retries submitted with `08:00:00`:

| Policy | Retry job | State |
|---|---|---|
| RBCBasicPolicy | e4363991-6cac-4cb4-a076-4980080ec4da | queued |
| RBCSmartPolicy | 0e481913-ae48-4f53-b34c-47beab830baa | queued |

`RBCCommunityPolicy` original is still running, so it has not been duplicated.
If it times out, submit the same `08:00:00` retry pattern.

## Active Long Runs

Current active/waiting rows:

| Wave | Policy | Seed | State | Notes |
|---|---|---:|---|---|
| wave2_baselines | RBCCommunityPolicy | - | running | CPU, original 2h allocation; may timeout |
| wave2_retry8h | RBCBasicPolicy | - | queued | retry after timeout |
| wave2_retry8h | RBCSmartPolicy | - | queued | retry after timeout |
| wave3_maddpg_direct | MADDPG_v3_direct | 123 | running | GPU, full-year, 6 episodes |
| wave3_maddpg_direct | MADDPG_v3_direct | 456 | queued | GPU |
| wave3_maddpg_direct | MADDPG_v3_direct | 789 | queued | GPU |
| wave4_maddpg_v48_teacher | MADDPG_v48_teacher | 123 | queued | GPU, teacher/BC recipe |
| wave4_maddpg_v48_teacher | MADDPG_v48_teacher | 456 | queued | GPU, teacher/BC recipe |
| wave4_maddpg_v48_teacher | MADDPG_v48_teacher | 789 | queued | GPU, teacher/BC recipe |

## Wave 1 Interpretation

Wave 1 was a 1024-step performance/profile wave, not a final scorecard wave.

Key reading:

```text
Deucalion GPU is the practical path for MADDPG training.

The selected full-year profile for Wave 3 is:
  update interval = 16
  batch size      = 256
  target          = Deucalion GPU
```

The Wave 1 CPU timeout is not being retried because it was only profiling and
the GPU result already answered the decision question.

## Next Actions

1. Poll Wave 2 retries and the running `RBCCommunityPolicy`.
2. If `RBCCommunityPolicy` times out, resubmit it with `08:00:00`.
3. Wait for Wave 3 seed 123 to finish or timeout before deciding whether to keep
   seeds 456/789 and Wave 4 queued as-is.
4. When any long MADDPG run finishes, rebuild:

```bash
.venv/bin/python scripts/build_phase8_v102_remote_scorecard.py \
  --summary-csv runs/remote_results/phase8_v102_wave1_final_2026_05_24/summary.csv \
  --summary-csv runs/remote_results/phase8_v102_wave2_latest_2026_05_24/summary.csv \
  --summary-csv runs/remote_results/phase8_v102_wave2_retries_8h_initial_2026_05_24/summary.csv \
  --summary-csv runs/remote_results/phase8_v102_wave3_now_2026_05_24/summary.csv \
  --summary-csv runs/remote_results/phase8_v102_wave4_latest_2026_05_24/summary.csv \
  --submitted-jobs runs/remote_configs/phase8_v102_wave1_2026_05_24/submitted_jobs.json \
  --submitted-jobs runs/remote_configs/phase8_v102_wave2_baselines_2026_05_24/submitted_jobs.json \
  --submitted-jobs runs/remote_configs/phase8_v102_wave2_retries_8h_2026_05_24/submitted_jobs.json \
  --submitted-jobs runs/remote_configs/phase8_v102_wave3_maddpg_direct_full_year_2026_05_24/submitted_jobs.json \
  --submitted-jobs runs/remote_configs/phase8_v102_wave4_maddpg_v48_teacher_full_year_2026_05_24/submitted_jobs.json \
  --output-dir runs/remote_results/phase8_v102_unified_scorecard_2026_05_24
```

