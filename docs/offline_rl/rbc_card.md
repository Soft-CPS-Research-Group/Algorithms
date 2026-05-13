# RBC Card — Behaviour Policy

> The Rule-Based Controller (RBC) is the **behaviour policy** that
> generates the dataset. This card describes *what it does*, *why
> it does it*, and *where it falls short* — the gaps that an offline
> RL agent has room to exploit.

Source: `algorithms/agents/rbc_agent.py` (class `RuleBasedPolicy`).

---

## 1. In one paragraph

The RBC is a hand-crafted controller for **EV charging only**. It looks
at each connected EV, asks "how much energy do I still need to deliver
before this car leaves?", and picks a charging rate that delivers that
energy on time, with a preference for charging when local solar
generation is available. It does **nothing** with the stationary
battery (`electrical_storage` action is hard-coded to 0). It has no
notion of grid price, carbon intensity, neighbourhood peak demand, or
ramping. It is purely reactive and per-charger.

---

## 2. Inputs (per agent, per step)

The RBC reads these observation fields (defaults in parentheses):

| Field | Purpose |
|------|---------|
| `electric_vehicle_charger_state` | Is a car plugged in? (0/1) |
| `electric_vehicle_soc` | Current state-of-charge (%) |
| `electric_vehicle_required_soc_departure` | Target SoC by departure (%) |
| `electric_vehicle_departure_time` | Hour-of-day the car leaves |
| `electric_vehicle_is_flexible` | Can charging be deferred? (0/1) |
| `hour`, `minute` | Current time |
| `solar_generation` *(or `electricity_generation` fallback)* | Local PV output |

Per-charger physical limits (max power, capacity, efficiency) come from
the dataset schema, not from observations.

---

## 3. Decision rules (in order)

For each EV slot:

1. **No car connected** → action = 0.
2. **Car connected and already at target SoC** (gap ≤ `energy_epsilon`)
 → action = 0.
3. **Compute "just enough" rate**
 `required_power = energy_needed / time_to_departure`
 then normalise: `n = min(1, required_power / max_power)`.
4. **PV bonus.** If `solar_generation ≥ pv_charge_threshold`
 (default 0), bump `n` up to at least `pv_preferred_charge_rate`
 (default 0.6). *Use the sun while it's free.*
5. **Emergency.** If `time_to_departure ≤ emergency_hours` (default 1 h)
 *or* the car is marked non-flexible, force `n` to at least
 `emergency_charge_rate` (default 1.0). *Don't strand the driver.*
6. **Trickle.** Otherwise, if flexible and no sun and plenty of time,
 trickle at `flex_trickle_charge` (default 0).
7. **Floor.** If `n > 0`, enforce `n ≥ min_charge_rate` (default 0).
8. **Clip** to the action-space bounds and emit.

The stationary-battery action is **always 0**.

### Key hyperparameters

| Name | Default | Meaning |
|------|---------|---------|
| `pv_charge_threshold` | 0.0 | PV level above which we prefer to charge from solar |
| `pv_preferred_charge_rate` | 0.6 | Floor on charge rate when solar is available |
| `flexibility_hours` | 3.0 | Beyond this, we can defer |
| `emergency_hours` | 1.0 | Below this, we top up at full power |
| `emergency_charge_rate` | 1.0 | Charge rate forced in emergencies |
| `flex_trickle_charge` | 0.0 | Background rate during flexible periods |
| `min_charge_rate` | 0.0 | Floor whenever we are charging at all |
| `energy_epsilon` | 1e-3 | "Already done" threshold |
| `default_capacity_kwh` | 60.0 | Fallback EV battery capacity |

---

## 4. What the RBC does well

- **Meets EV deadlines.** By construction, `required_power` is computed
 to deliver `energy_needed` by `departure_time`. Drivers do not get
 stranded.
- **Uses local PV.** Charging is biased toward periods of high solar
 output, reducing grid imports during the day.
- **Stable and deterministic.** Same inputs → same actions; trivially
 reproducible.

---

## 5. What the RBC does *not* do (room for improvement)

These are the gaps an offline RL agent can target. Each gap maps to one
or more CityLearn KPIs (see `kpi_reference.md`).

| Gap | KPI(s) it hurts |
|-----|----------------|
| **No price awareness.** Doesn't shift charging to cheap hours. | `cost_total` |
| **No carbon awareness.** Doesn't shift charging to low-carbon hours. | `carbon_emissions_total` |
| **No neighbourhood peak shaving.** Per-charger logic ignores district demand; multiple chargers can pull together and create peaks. | `daily_peak_average`, `all_time_peak_average` |
| **No ramping control.** Sudden on/off transitions when sun appears or emergencies trigger. | `ramping_average` |
| **No load-factor smoothing.** Charging is bursty (PV bonus + emergency cliff) rather than spread. | `daily_one_minus_load_factor_average` |
| **Stationary battery unused.** Action hard-coded to 0 → forfeits arbitrage and self-consumption. | `cost_total`, `electricity_consumption_total`, `zero_net_energy` |
| **No anticipation.** Reacts to *current* solar, not forecasted solar/price/carbon (forecast columns exist in the obs but are ignored). | All of the above |
| **No V2G discharge.** EV battery is never used to support the building or grid. | `cost_total`, `daily_peak_average` |

In short: the RBC is a reasonable EV scheduler and a *trivial* battery
controller. The offline RL agent inherits the EV competence (via
imitation in BC, via dataset coverage in IQL) and gets the chance to
**add** price/carbon/peak/ramping awareness and **activate** the
stationary battery.

---

## 6. Why this matters for 

- **Dataset diversity.** The RBC's behaviour is mostly deterministic
 given the inputs, so multi-seed rollouts give us *environmental*
 diversity (EV arrival times, weather, prices) rather than *action*
 diversity. That's still useful for state coverage, but it caps the
 best policy IQL can extract by what the RBC happens to explore.
 → If IQL plateaus at RBC level, the next iteration must use a
 noisier or learned behaviour policy to broaden action coverage.
- **Reward design.** The RBC's blind spots (peak, ramping, price,
 carbon, idle battery) tell us exactly which terms the reward must
 include if we want the agent to do better than imitate. See
 `reward_design.md`.
- **Evaluation.** Any agent that ties RBC on EV-deadline metrics but
 improves on `cost_total`, `carbon_emissions_total`, `peak`, or
 `ramping` is a win.
