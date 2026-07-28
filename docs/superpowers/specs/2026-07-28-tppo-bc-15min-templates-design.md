# TransformerPPO BC 15-Minute Templates Design

## Goal

Provide one fast local smoke template and three reproducible server-training
templates for `AgentTransformerPPO` behavior cloning on the dynamic 15-minute
dataset.

## Dataset

All templates use:

`datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json`

The dataset has 35,040 15-minute steps (one non-leap year) and dynamic topology
events beginning with a charger addition at step 5,200.

## Templates

### Smoke

The smoke template reads dataset steps 5,184 through 5,248. CityLearn resets
its runtime clock to zero for a sliced episode, so the template applies
`topology_event_time_offset: -5184` when loading the local schema. This shifts
the real charger-add event from dataset step 5,200 to episode-local step 16.
The 65-step window is long enough for multiple 16-step PPO updates. The smoke
succeeds only if:

- the real dataset and wrapper initialize;
- the RBC teacher produces behavior-cloning targets;
- at least one PPO update reports finite BC diagnostics; and
- `topology_version` changes and the Transformer layout rebuilds.

### Server Experiments

The server templates each run one full-year episode. The installed simulator
does not restore dynamically added or removed assets when an episode resets, so
multi-episode runs would not replay the same topology schedule. A single
full-year episode is the only reproducible dynamic-topology configuration until
that simulator reset behavior is fixed.

The templates differ only in teacher duration:

| Template | BC decay / action phaseout |
|---|---:|
| week | 672 steps (7 days) |
| month | 2,880 steps (30 days) |
| year | BC loss: 34,816 (final PPO update); blend: 35,039 decisions |

All variants decay BC loss from `0.42` to `0.0` within the executable update
schedule. The year variant uses the final scheduled PPO update (34,816) for BC
loss and the final action decision (35,039) for blending. They use the same model,
dataset, reward, seed, update cadence, and full-year horizon so server runs are
directly comparable. Week and month variants spend the remainder of the year
as pure PPO; the year variant remains teacher-guided for the full run.

## Runtime Expectation

The templates do not impose a 10-hour wall-clock limit or guarantee a minimum
runtime. Runtime depends on server hardware. Measured throughput from the first
jobs determines whether a separate simulator topology-reset fix is needed to
support reproducible multi-episode runs longer than one year.

## Scope

This work adds configuration, verification, and a narrowly scoped local-schema
topology-event offset used by the smoke window. It does not change the BC
algorithm, Transformer architecture, dataset, or reward implementation.
