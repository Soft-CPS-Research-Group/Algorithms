# Offline RL for Energy Communities — Project Status

## 1. The Problem

Renewable Energy Communities (RECs) bring together buildings, solar panels, and electric
vehicles into a shared energy system. Managing when and how to charge EVs is critical:
charge at the wrong time and the community pays more, stresses the grid, and wastes
renewable energy. The ideal solution learns a smart charging strategy that reacts to
electricity prices, solar availability, and each vehicle's schedule.

The real-world data used throughout this project comes from the **OPEVA dataset** — a
Portuguese REC consisting of 17 residential and commercial buildings with 8 EV charging
stations, covering August 2022 to July 2023 (one full year). Data was collected at hourly
resolution in the field and is simulated at 15-second resolution, capturing fine-grained
energy dynamics across the community.

The straightforward approach — let an AI agent learn by trial and error in the real system
— is not viable. Bad decisions carry real costs, grid operators impose constraints, and
running a year-long simulation for every training attempt is slow and expensive.

**Offline reinforcement learning** sidesteps this: instead of exploring live, the agent
learns entirely from a historical log of past behaviour. No interaction with the real system
is needed during training. The learned policy is only deployed (or simulated) once it is
ready.

## 2. The Journey

### Phase 1 — Proof of Concept

The first step was to prove the idea works at all. Two buildings from the OPEVA dataset
were selected, one month of hourly data was used, and a single IQL agent was trained in a
Python notebook using an existing offline RL library.

The agent matched the rule-based expert on Building 1 and slightly outperformed it on
Building 3 without any retraining — an early sign that the learned policy could generalise
across buildings it had never seen.

This phase established confidence in the approach and defined the research direction.

### Phase 2 — Scaling to the Full Community

The notebook was replaced with a proper pipeline covering all 17 buildings in the OPEVA
community (including 8 EV charging stations) at 15-second time resolution — the full
resolution of the real dataset.

The key design challenge was heterogeneity: buildings have different numbers of EV
chargers, different observation sizes, and different action spaces. Training one model per
building would be expensive and would throw away the similarity between buildings of the
same type.

The solution was **agent grouping**: buildings with identical structure share a single
trained model. Four groups were identified automatically from the dataset. This meant
training 4 models instead of 17, with each model covering all buildings of its type.

A **feature analysis** was also conducted at this stage: the dataset was examined to
understand which observations are most informative, how EV charging patterns look across
the day, which features are redundant, and what the data says about charging urgency.
This informed the design of how observations are encoded before reaching the agents.

### Phase 3 — Two Algorithms, One Comparison

IQL was joined by a second algorithm, **CQL (Conservative Q-Learning)**, which takes a
more cautious approach: it actively suppresses confidence in actions the expert never
demonstrated. Both were trained on the same dataset and evaluated against the same
rule-based expert baseline across 5 independent evaluation scenarios.

This produced the first rigorous, apples-to-apples comparison of the two approaches on a
real-scale community.

### Phase 4 — Understanding the Results

After the benchmark, a behavioural analysis was conducted: rather than just looking at
summary numbers, the per-step decisions of all three policies were logged and plotted.
This revealed *why* the results differ — how charging is distributed across the day, how
each policy responds to electricity prices, and what the demand profile looks like for the
whole community.

## 3. How the Pipeline Works

Data is collected first by running the rule-based expert controller across 10 independent
scenarios, saving every observation, action, and reward to disk. The feature analysis tool
then inspects this data to validate quality and understand patterns. Four AI models are
trained — one per building type — each learning from the relevant slice of the dataset.
At evaluation time, the 4 models are assembled into a single community-level agent: an
adapter layer routes each building's observations to its corresponding group model and
collects all actions back into one response for the simulator. This assembled agent is then
evaluated in 5 fresh scenarios it has never seen, and its decisions are compared step-by-step
against the rule-based expert and the second AI algorithm.

## 4. Results

Evaluated across 5 independent scenarios, against the rule-based expert (RBCSmart):

| Metric             | IQL        | CQL        |
|--------------------|------------|------------|
| Electricity cost   | **−21.5%** | −18.3%     |
| Carbon emissions   | **−14.7%** | −12.0%     |
| Peak demand        | **−29.2%** | −22.2%     |
| Grid ramping       | −32.5%     | **−54.3%** |
| Self-sufficiency   | Worse      | Worse      |

Both AI agents substantially outperform the rule-based expert on cost, carbon, and peak
demand. IQL leads on savings; CQL leads on grid stability. Self-sufficiency regresses for
both — see Key Insights below.

## 5. Key Insights

**One model per building type covers the whole community.**
Training 4 shared models instead of 17 individual ones was not a compromise — it worked
because buildings of the same structural type live in the same observation space. The
approach scales to larger communities without architectural changes.

**Both agents learned a strategy the rule-based expert never used.**
The expert charges EVs reactively, triggered by connection events. Both AI agents learned
to shift charging to overnight low-price windows — a behaviour that is latent in the data
but never made explicit. This is what drives the cost and carbon reductions.

**IQL saves money; CQL protects the grid.**
IQL is willing to charge aggressively when prices are low, producing abrupt load swings.
CQL, being more conservative, produces smoother charging profiles. The right choice depends
on the deployment context: if grid ramping limits are binding, CQL is preferable; if cost
reduction is the primary goal, IQL wins.

**Self-sufficiency got worse — and that was expected.**
Both agents import more electricity from the grid than the expert. This is a direct
consequence of how they were trained: the reward signal optimises for cost, not
self-sufficiency. Charging at off-peak hours (cheap, low-carbon) coincides with low solar
generation, so more grid import is the rational strategy given the objective. Fixing this
requires a reward that explicitly values renewable self-consumption.

**The dataset shapes what agents can learn.**
The expert controller never uses stationary battery storage — it only manages EV charging.
As a result, the collected dataset contains no examples of battery charge/discharge
behaviour. The agents therefore never learned to use the battery, even though it is
physically present. Better behaviour policies collect richer, more diverse data.

## 6. How to Replicate

The full pipeline runs in sequence. Each step builds on the previous one.

**Step 1 — Collect the dataset**
Run the rule-based expert controller in the CityLearn simulation across 10 scenarios
(9 for training, 1 for validation). Observations, actions, and rewards are saved to disk
as structured files, one per building type.

**Step 2 — Analyse the dataset (optional but recommended)**
Run the feature analysis tool to inspect the collected data: check that scenarios are
statistically consistent, examine which observations carry the most information, and
understand EV charging patterns across the day.

**Step 3 — Train IQL**
Run the IQL training script. One model is trained per building type (4 runs total).
Training takes approximately 3–4 hours per group on a standard CPU. The best checkpoint
per group is selected automatically based on validation performance on the held-out
scenario.

**Step 4 — Train CQL**
Same as Step 3, using the CQL training script. The same dataset and building groups are
used; only the algorithm differs.

**Step 5 — Run the benchmark**
The benchmark script loads all trained models and the rule-based expert, rolls out one full
episode per policy per evaluation scenario (5 scenarios), and produces a summary table of
KPIs.

**Step 6 — Generate behavioural analysis figures**
The analysis script replays one evaluation episode for all three policies and generates
four diagnostic figures: community demand profile, EV action patterns by hour, price
responsiveness, and KPI comparison.

## 7. Summary

### What Was Done

- Collected a full offline training dataset from the OPEVA REC (17 buildings, 8 EV
  stations, one year at 15-second resolution) by running a rule-based expert controller
  across 10 independent scenarios.
- Conducted a feature analysis on the collected data: validated seed consistency, identified
  the most informative observations, mapped EV charging patterns across the day, and
  proposed derived features for future training.
- Designed an observation encoding system that correctly handles time (circular encoding),
  categorical states (one-hot), and continuous measurements (normalisation with missing
  value handling).
- Trained an IQL agent — optimising for electricity cost minimisation — across all 4
  building type groups, sharing one model per group to cover all 17 buildings efficiently.
- Trained a CQL agent — using a conservative approach that avoids overconfident decisions
  — under the same conditions for a direct comparison.
- Assembled the 4 trained group models into a single community-level agent using an adapter
  layer that routes each building's observations to its corresponding model and collects
  all actions back into a unified response for the simulator at evaluation time.
- Evaluated all three policies (IQL, CQL, rule-based expert) across 5 independent
  scenarios using 5 KPIs: electricity cost, carbon emissions, peak demand, grid ramping,
  and self-sufficiency (Zero Net Energy).
- Generated behavioural analysis figures showing how each policy distributes EV charging
  across the day, how it responds to electricity prices, and how community demand differs
  across the three strategies.

### Key Findings

- **IQL reduced electricity cost by 21.5% and peak demand by 29.2%** relative to the
  rule-based expert, measured as the mean improvement across 5 evaluation scenarios.
- **CQL reduced grid ramping by 54.3%**, producing the smoothest charging profiles of the
  three policies — a significant advantage for grid stability.
- **Both algorithms outperformed the rule-based expert on all cost, carbon, and peak KPIs**,
  demonstrating that offline RL can learn strategies that a hand-crafted rule controller
  does not capture.
- **A meaningful trade-off exists between IQL and CQL**: IQL is preferable when cost
  reduction is the primary goal; CQL is preferable when grid stability is a binding
  constraint. The choice depends on deployment context.
- **Self-sufficiency (ZNE) worsened for both AI agents** because the reward function
  optimises cost, not renewable self-consumption. Shifting load to cheap off-peak hours
  coincides with low solar generation. This is an intentional consequence of the reward
  design, not a failure of the algorithms.
- **One model per building type is sufficient to manage the full community.** The 4 group
  models were assembled seamlessly into a single community agent, covering all 17 buildings
  without per-building retraining. This confirms the scalability of the approach.

### Future Work

- **Extend evaluation to 10 scenarios** for tighter statistical confidence in the results.
- **Systematically tune algorithm settings** (e.g. how aggressively IQL exploits the data,
  how conservatively CQL penalises uncertainty) to improve performance across all building
  groups.
- **Add derived features** from the feature analysis — in particular a charging urgency
  score combining remaining charge deficit with time until departure — which could improve
  convergence speed and final performance.
- **Collect a richer dataset** using a behaviour policy that also demonstrates stationary
  battery usage, enabling AI agents to learn battery charge/discharge strategies currently
  absent from the training data.
- **Redesign the reward to balance cost and self-sufficiency**, preventing the ZNE
  regression and pushing agents toward strategies that benefit both the community's energy
  bill and its independence from the grid.
- **Test the pipeline on a second, independent community** without retraining, to confirm
  that the approach generalises beyond the OPEVA dataset.
- **Add safety constraints** ensuring EVs always reach the required charge level before
  departure — a prerequisite for any real-world deployment.
- **Connect the pipeline to a live energy management system**, replacing simulation with
  real sensor data and real charging commands, as the final step toward deployment in an
  actual REC.
