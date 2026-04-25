# Offline Reinforcement Learning — Q&A

---

## 1. Does an ORL model need a dataset to be trained on?

**Yes, that is the defining characteristic of Offline RL (also called Batch RL).**

In standard (online) RL the agent learns by interacting with the environment in real time: it takes an action, observes the outcome, and updates its policy. In Offline RL there is **no interaction with the environment during training**. Instead, the agent learns entirely from a **fixed, previously collected dataset** of transitions.

This dataset is typically gathered by one or more **behaviour policies** — for example a rule-based controller, a human operator, or even a random policy. The quality, diversity, and coverage of this dataset are critical: the agent can only learn about state–action regions that are represented in the data.

Empirically, researchers have shown that Offline RL can match or even outperform the behaviour policy that collected the data, but only when:
- the dataset has reasonable coverage of the state–action space, and
- the algorithm includes some mechanism to stay close to the data distribution (to avoid overestimating the value of unseen actions).

---

## 2. What must the dataset contain?

A standard Offline RL dataset is a collection of **transitions** (also called tuples). Each transition contains:

| Field | Symbol | Description |
|---|---|---|
| **Observation (state)** | $s_t$ | The state of the environment at time $t$ |
| **Action** | $a_t$ | The action taken by the behaviour policy |
| **Reward** | $r_t$ | The scalar reward received after taking $a_t$ in $s_t$ |
| **Next observation** | $s_{t+1}$ | The state the environment transitioned to |
| **Terminated flag** | $d_t$ | Whether the episode ended (terminal state reached) |
| **Truncated flag** *(optional)* | $tr_t$ | Whether the episode was cut short by a time limit (not a true terminal state) |

So the minimal tuple is $(s_t, a_t, r_t, s_{t+1}, d_t)$.

**Episode duration** is not stored explicitly as a field; it is implicitly captured by the sequence of transitions and the terminated/truncated flags that mark episode boundaries.

Additional metadata that is **useful but not strictly required**:

- **Episode index** — to group transitions by episode (helpful for sequence-modelling approaches like Decision Transformer).
- **Timestep within episode** — again useful for transformer-based methods.
- **Behaviour policy identifier** — if data comes from multiple policies, knowing which policy produced each transition can help with importance-weighting or filtering.
- **Action probabilities / log-probabilities** — needed by some off-policy correction methods (e.g., importance sampling), but not by most modern Offline RL algorithms (CQL, IQL, TD3+BC, etc.).

---

## 3. Must the observations in the dataset match the observations used by the behaviour policy?

**Yes — the observations recorded in the dataset must be the same observations that were fed to the behaviour policy when it chose the action.**

The reason is causal consistency: the tuple $(s_t, a_t, r_t, s_{t+1})$ must faithfully represent what the behaviour policy saw ($s_t$), what it decided ($a_t$), what the environment returned ($r_t$), and where the environment landed ($s_{t+1}$). If you record a different observation than what the policy actually used, the relationship between state and action becomes inconsistent, and the learned policy will be unreliable.

**However**, the Offline RL agent you train **does not have to use the exact same observation representation**. You are free to:

- **Add features** — e.g., append time-of-day, day-of-week, or rolling averages that were not used by the behaviour policy. As long as these features are computable from the raw environment state, they are valid.
- **Remove features** — you can train your new agent on a subset of the recorded observations (see Question 4).
- **Transform features** — normalisation, discretisation, embedding, etc.

The key constraint is on the **data collection** side: record what the behaviour policy actually saw and did. On the **training** side, you have flexibility.

---

## 4. Is feature engineering relevant for Offline RL?

**Yes, feature engineering is relevant and can be quite impactful — but you are right to distinguish between two separate concerns.**

### Concern A: The dataset generation side

When you generate the dataset, the behaviour policy (e.g., your rule-based controller or MADDPG agent) requires a specific set of observations to produce actions. **You cannot remove observations that the behaviour policy needs** during data collection — that would break the policy. So on the data-collection side, you record everything the behaviour policy consumes.

### Concern B: The Offline RL training side

When training your Offline RL agent, you **can** choose which features to use. You are building a **new** policy that learns from $(s, a, r, s')$ tuples. This new policy can use any subset or transformation of $s$. Common practices:

| Technique | Example | Empirical benefit |
|---|---|---|
| **Feature selection** | Drop features with near-zero variance or high correlation | Reduces overfitting, especially on small datasets |
| **Feature importance analysis** | SHAP values, permutation importance on a fitted Q-function | Helps interpretability and can guide pruning |
| **Normalisation** | Z-score or min–max scaling | Almost always improves training stability |
| **Temporal features** | Add lag features, rolling means | Captures dynamics the raw observation may miss |

### Your intuition is correct

Feature importance analysis is valuable for **explanation and understanding** (e.g., "electricity price is 3× more influential than outdoor temperature for charging decisions"), even if you ultimately keep all features for training. In an academic thesis, this kind of analysis is highly relevant — it shows you understand what the agent is learning and why.

Empirically, for modest-dimensional observation spaces (like CityLearn's ~20–50 features per building), removing features rarely gives dramatic performance gains, but it does improve interpretability and can reduce overfitting when data is limited.

---

## 5. Do different ORL algorithms require different datasets?

**No — the same dataset of $(s, a, r, s')$ tuples can be used by any Offline RL algorithm.** The dataset format is algorithm-agnostic.

What changes across algorithms is:

| Aspect | How it varies |
|---|---|
| **How the data is consumed** | Some algorithms sample random mini-batches (CQL, IQL, TD3+BC); others process full episodes sequentially (Decision Transformer, Trajectory Transformer). |
| **What the algorithm learns** | Actor-critic methods learn a Q-function and a policy; sequence models learn to predict actions given a desired return. |
| **Constraint mechanism** | CQL penalises out-of-distribution actions via a conservative Q-estimate; TD3+BC adds a BC (behaviour cloning) regulariser; IQL avoids querying unseen actions entirely. |
| **Sensitivity to data quality** | Simpler methods (BC, TD3+BC) are more robust to small/noisy datasets; more complex methods (CQL) can extract more value from larger, diverse datasets but are harder to tune. |

### Practical recommendation for your thesis

Start with a simple, well-understood algorithm:

1. **Behaviour Cloning (BC)** — pure supervised learning baseline. Train a policy to imitate the behaviour policy. Fast, easy to debug, sets a floor.
2. **TD3+BC** — a minimal modification of TD3 that adds a behaviour cloning term. Very easy to implement and tune (one extra hyperparameter $\alpha$).
3. **IQL (Implicit Q-Learning)** — avoids querying out-of-distribution actions entirely. Considered one of the most stable Offline RL methods empirically.

Once comfortable, you can explore **CQL** or **Decision Transformer** with the **same dataset** — no need to recollect data.

---

## 6. Will the final ORL model return actions in the same format as the behaviour policy?

**Yes.** The Offline RL agent learns a policy $\pi(a|s)$ that maps observations to actions. Since it was trained on actions produced by the behaviour policy, it outputs actions in **the same action space** — same dimensionality, same ranges, same semantics.

For example, if the behaviour policy outputs a continuous action in $[-1, 1]$ representing a charging/discharging rate for each building, the trained Offline RL agent will also output a value in $[-1, 1]$ for each building.

The **values** of the actions will generally be different (that is the whole point — you hope the new policy makes *better* decisions), but the **format** is identical. This means the trained agent can be deployed as a drop-in replacement for the behaviour policy in the CityLearn environment or in the existing wrapper infrastructure.

---

## 7. How should the ORL model be evaluated? Should we compare against the behaviour policy?

### Using CityLearn KPIs

Since you have a working CityLearn wrapper, you can evaluate any trained agent by running it in the environment and collecting the platform's built-in KPIs (electricity cost, carbon emissions, grid stability, etc.). This is the most meaningful evaluation because it measures real task performance.

### What to compare against

You should compare your Offline RL agent against **multiple baselines**:

| Baseline | Why |
|---|---|
| **Behaviour policy** (the policy that collected the data) | The most direct comparison — did Offline RL improve over the data source? |
| **Behaviour Cloning (BC)** | Shows whether the Offline RL algorithm adds value beyond simple imitation. If BC matches Offline RL, the RL component is not contributing. |
| **Random policy** | A sanity-check lower bound. |
| **Rule-based controller (RBC)** | If your behaviour policy is MADDPG, comparing against RBC shows the value chain: RBC → MADDPG → Offline RL. |
| **No-action baseline** | Buildings with no active control — shows the value of any control at all. |

### Evaluation metrics

Beyond CityLearn KPIs, standard Offline RL evaluation metrics include:

| Metric | Description |
|---|---|
| **Normalised score** | $\frac{\text{agent return} - \text{random return}}{\text{expert return} - \text{random return}} \times 100$. Widely used in D4RL benchmarks. Gives a 0–100+ scale. |
| **Average episodic return** | The mean cumulative reward across evaluation episodes. |
| **Return distribution** | Not just the mean — plot the distribution (box plots, histograms) to assess consistency. |
| **Per-building KPIs** | Since CityLearn is multi-agent, break down performance per building to spot agents that underperform. |

### Empirical tip

Run evaluation over **multiple seeds** (at least 3–5) and report mean ± standard deviation. Offline RL results can be sensitive to initialisation.

---

## 8. What data augmentation techniques can be applied to Offline RL?

Data augmentation in Offline RL is an active research area. Here are techniques that are empirically validated and relevant to your setting:

### 8.1 State-based augmentations

| Technique | Description | When useful |
|---|---|---|
| **Gaussian noise injection** | Add small $\mathcal{N}(0, \sigma^2)$ noise to observations (S-O4RL, RADA). | General-purpose; improves robustness. $\sigma$ should be small relative to feature scale. |
| **Feature dropout / masking** | Randomly zero out a fraction of observation features during training. | Acts as regularisation; prevents over-reliance on single features. |
| **Observation interpolation (Mixup)** | Linearly interpolate between two transitions: $s' = \lambda s_1 + (1-\lambda) s_2$. | Can improve generalisation, but reward interpolation must be handled carefully. |

### 8.2 Transition-level augmentations

| Technique | Description | When useful |
|---|---|---|
| **Reward perturbation** | Add small noise to rewards. | Can reduce overfitting to reward signal quirks. |
| **Trajectory stitching** | Combine sub-trajectories from different episodes to create new plausible trajectories. | Useful when coverage is sparse — creates "shortcuts" through state space. |
| **Hindsight Experience Replay (HER)** | Re-label transitions with alternative goals/rewards. | Applicable if the task has a goal-conditioned structure. |

### 8.3 Model-based augmentation

| Technique | Description | When useful |
|---|---|---|
| **Learned dynamics model** | Train a model $\hat{s}_{t+1} = f(s_t, a_t)$ on the dataset, then generate synthetic transitions. | Most powerful but also most complex. Risk of compounding model errors. |
| **COMBO / MOReL** | Full model-based Offline RL methods that use a learned model to augment the dataset with pessimistic or conservative rollouts. | When the dataset is small and you need more coverage. |

### Practical recommendation

For a thesis, a clean experiment would be:

1. **Baseline**: Train Offline RL on the original dataset.
2. **+Gaussian noise**: Train on the augmented dataset with $\sigma \in \{0.01, 0.05, 0.1\}$.
3. **+Mixup**: Train with state–action interpolation.
4. Compare all three on CityLearn KPIs.

This gives you a concrete empirical contribution ("data augmentation effect on Offline RL for building energy management") without excessive complexity.

---

## 9. Additional considerations

### 9.1 Dataset quality matters more than algorithm choice

This is the single most important empirical finding in Offline RL research. In the D4RL benchmark (Fu et al., 2020), the same algorithm can go from 20% to 95% normalised score just by switching from a "random" dataset to a "medium-expert" dataset. Invest time in collecting good data:

- Use a **reasonably good** behaviour policy (your trained MADDPG or a tuned RBC).
- Ensure the dataset covers **diverse conditions** (different seasons, demand patterns, EV arrival/departure times).
- Include some **sub-optimal transitions** — pure expert data can cause brittleness because the agent never sees recovery from mistakes.

A mixed dataset (e.g., 70% from a good policy + 30% from a noisier/exploratory policy) often works best empirically.

### 9.2 Conservative / pessimistic learning is key

The central challenge of Offline RL is **distributional shift**: the agent may learn to take actions that look good according to an extrapolated Q-function but are actually terrible because they were never tried in the data. All successful Offline RL algorithms address this via some form of pessimism or conservatism. When debugging poor results, the first thing to check is whether the agent is selecting out-of-distribution actions.

### 9.3 Hyperparameter sensitivity

Offline RL algorithms have hyperparameters that control the conservatism–optimality trade-off (e.g., $\alpha$ in CQL, $\alpha$ in TD3+BC, $\tau$ in IQL). These require tuning and the optimal value depends on the dataset. Since you cannot tune online (no environment interaction), use:

- **Validation on held-out episodes** from the dataset.
- **Policy evaluation estimators** (e.g., Fitted Q Evaluation) to estimate return without running the policy.

### 9.4 Multi-agent considerations

CityLearn is a multi-agent environment. You have two main design choices for Offline RL:

| Approach | Description | Trade-off |
|---|---|---|
| **Independent learners** | Train one Offline RL agent per building, each seeing only its own observations. | Simple, scales well, but ignores inter-building coordination. |
| **Centralised training** | Train a single agent that sees all buildings' observations and outputs all actions. | Can capture coordination, but action/observation space grows with number of buildings. |

For a first pass, independent learners are simpler and often work surprisingly well.

### 9.5 Libraries and frameworks

For implementation, consider:

| Library | Algorithms | Notes |
|---|---|---|
| **d3rlpy** | CQL, IQL, TD3+BC, BC, Decision Transformer | Pure Python, clean API, active maintenance. Probably your best bet. |
| **CORL** | CQL, IQL, TD3+BC, SAC-N, EDAC, DT | Clean single-file implementations, good for understanding. |
| **CleanRL** | Online RL + some offline | Good for learning, less Offline RL coverage. |

`d3rlpy` is particularly well-suited because it can load datasets in a simple numpy format (observations, actions, rewards, terminals as arrays), which aligns directly with what you'd record from CityLearn.

---

*Document created: 13 April 2026*
