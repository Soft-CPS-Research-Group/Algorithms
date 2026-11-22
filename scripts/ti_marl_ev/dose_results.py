"""Save compact results and independently reloadable, isolated-EV research actors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.ti_marl.experiments.ev_pretraining import FORMAT, load_ev_policy
from scripts.train_ti_marl_ev_sessions import SessionProbe, play

from .dose_evaluate import bands, paired, single
from .from_scratch import fresh
from .intermediate import OUT, ROOT, sealed_cases
from .native_curves import original_curve_sessions
from .protocol import TRACE, split

SEEDS = (43, 73)
WIDTHS = (32, 64, 128)


@torch.no_grad()
def export(third_only=False):
    torch.set_num_threads(1)
    selection = json.loads((OUT / "selection.json").read_text())
    dest = OUT / "models"
    dest.mkdir(exist_ok=True)
    base = yaml.safe_load((ROOT / "configs/templates/rl/ti_marl_core30.yaml").read_text())
    _, dev, _ = split()
    probe = SessionProbe(TRACE)
    verification = json.loads((OUT / "reload.json").read_text()) if third_only else {}
    for w in WIDTHS:
        method, step = (selection[str(w)]["selected"][k] for k in ("method", "step"))
        for s in (109,) if third_only else SEEDS:
            a = fresh(w, s)
            a.actor.eval()
            weights = torch.load(
                OUT / f"{method}-{w}-{s}-{step}.pth", map_location="cpu", weights_only=True
            )["actor"]
            a.actor.load_state_dict(weights)
            parameters = dict(
                base["pipeline"][0]["hyperparameters"]["actor"], d_model=w, variant="v2"
            )
            path = dest / f"v2-{w}-{s}.pth"
            torch.save(
                dict(
                    format=FORMAT,
                    adapter="wide",
                    type_registry=a.compiler.type_registry,
                    actor_parameters=parameters,
                    weights=weights,
                    operating_parameters={
                        k: getattr(a.actor, k)
                        for k in (
                            "step_seconds",
                            "departure_margin_seconds",
                            "tolerance_soc",
                            "order_seed",
                        )
                    },
                    training=dict(
                        teacher=False,
                        policy="continuous_interior",
                        incremental_excess_eur_per_kwh=0.05,
                        seed=s,
                        recipe=method,
                        updates_in_final_phase=step,
                    ),
                ),
                path,
            )
            reference = [play(z, a, None, probe, "eval")["kwh"] for z in dev[:12]]
            a.actor = load_ev_policy(path)
            replay = [play(z, a, None, probe, "eval")["kwh"] for z in dev[:12]]
            error = max(
                abs(x - y) for left, right in zip(reference, replay) for x, y in zip(left, right)
            )
            assert error < 1e-7
            verification[f"{w}-{s}"] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decisions=sum(map(len, replay)),
                max_kwh_difference=error,
            )
    (OUT / "reload.json").write_text(json.dumps(verification))


@torch.no_grad()
def equal_energy(width, seed, output):
    torch.set_num_threads(1)
    actor = load_ev_policy(OUT / "models" / f"v2-{width}-{seed}.pth")
    cases = original_curve_sessions(
        actor, include_cost_reference=False, conditions=sealed_cases()["fresh"], variable_need=True
    )
    result = {}
    for hour in (8, 16):
        subset = [r for r in cases if r["hour"] == hour]
        result[hour] = {
            field: sum(r[field] for r in subset)
            for field in (
                "applied",
                "requested",
                "cost",
                "equal_energy_immediate_cost",
                "solar_kwh",
                "equal_energy_immediate_solar",
            )
        }
        result[hour]["dates"] = len(subset)
        result[hour]["cost_lower_dates"] = sum(
            r["cost"] < r["equal_energy_immediate_cost"] - 1e-8 for r in subset
        )
    output.write_text(json.dumps(result, allow_nan=False))


@torch.no_grad()
def learning_curves(output):
    """After freezing model choices, replay predetermined durations on the 32 new dates."""
    torch.set_num_threads(1)
    cases = sealed_cases()["fresh"]
    result = {}
    for width in WIDTHS:
        for seed in SEEDS:
            agent = fresh(width, seed)
            agent.actor.cpu().eval()
            for method, stages in (
                ("gradient", (3072, 12288, 49152)),
                ("fork", (256, 1024, 4096)),
                ("polish", (256, 1024, 2048, 4096)),
            ):
                for stage in stages:
                    tag = f"{method}-{width}-{seed}-{stage}"
                    weights = torch.load(OUT / f"{tag}.pth", map_location="cpu", weights_only=True)[
                        "actor"
                    ]
                    agent.actor.load_state_dict(weights)
                    m = paired(cases, agent, None, True)
                    dose = single(bands(cases), agent, None, True)
                    result[tag] = dict(
                        sessions=m["sessions"],
                        within=m["near_target"],
                        mae=m["mae_kwh"],
                        price=m["price"]["timing_and_target"],
                        solar=m["solar"]["timing_and_target"],
                        dose=dose["within"],
                    )
    output.write_text(json.dumps(result))


def combine(reps):
    ans = {}
    for k in ("previous_round", "previous_fractional", "fresh"):
        xs = [r[k] for r in reps]
        summed = (
            "sessions",
            "near_target",
            "target_reached",
            "protected",
            "charged_after_target",
            "decisions_after_target",
            "short_by_over_0_1",
            "over_by_over_0_1",
        )
        avg = ("mae_kwh", "mean_extra_kwh", "mean_missing_kwh", "bill_eur")
        m = {key: sum(x[key] for x in xs) for key in summed if key in xs[0]}
        m.update({key: float(np.mean([x[key] for x in xs])) for key in avg})
        for key in ("worst_shortfall", "worst_excess"):
            if key in xs[0]:
                m[key] = max(x[key] for x in xs)
        for v in ("price", "solar"):
            m[v] = {
                key: sum(x[v][key] for x in xs) for key in ("pairs", "timing", "timing_and_target")
            }
            for key in ("mean_shift_kwh", "in_window_share", "from_solar_share"):
                if key in xs[0][v]:
                    m[v][key] = float(np.mean([x[v][key] for x in xs]))
        ans[k] = m
    for k in ("original", "fresh_curves"):
        ans[k] = {}
        for h in ("8", "16"):
            xs = [r[k][h] for r in reps]
            ans[k][h] = {
                v: sum(x[v] for x in xs)
                for v in (
                    "n",
                    "within",
                    "bill",
                    "immediate_bill",
                    "matched_bill",
                    "matched_immediate",
                    "solar",
                    "immediate_solar",
                    "delivered",
                    "requested",
                )
            }
            ans[k][h].update({v: float(np.mean([x[v] for x in xs])) for v in ("mae", "mean_error")})
            ans[k][h].update(
                {v: max(x[v] for x in xs) for v in ("worst_shortfall", "worst_excess")}
            )
    for k in ("dose", "stop"):
        xs = [r[k] for r in reps]
        ans[k] = dict(
            n=sum(x["n"] for x in xs),
            within=sum(x["within"] for x in xs),
            mae=float(np.mean([x["mae"] for x in xs])),
            max_error=max(x["max_error"] for x in xs),
            example_075=[x["example_075"] for x in xs],
        )
    return ans


def confirmation(reports):
    result = {}
    seeds = (*SEEDS, 109)
    selection = json.loads((OUT / "service-selection.json").read_text())
    for phase in ("prior", "candidate", "router"):
        for w in WIDTHS:
            trials = {
                s: json.loads((reports / f"final-{phase}-{w}-{s}.json").read_text()) for s in seeds
            }
            rows = list(trials.values())
            groups = {"per_seed": trials}
            for cohort in ("new", "regression") if phase != "prior" else ("new",):
                xs = [r[cohort] for r in rows]
                groups[cohort] = dict(
                    n=sum(x["sessions"] for x in xs),
                    within=sum(x["near_target"] for x in xs),
                    mae=float(np.mean([x["mae_kwh"] for x in xs])),
                    short_by_over_0_1=sum(x["short_by_over_0_1"] for x in xs),
                    over_by_over_0_1=sum(x["over_by_over_0_1"] for x in xs),
                    worst_shortfall=max(x["worst_shortfall"] for x in xs),
                    worst_excess=max(x["worst_excess"] for x in xs),
                    **{k: sum(x[k]["timing_and_target"] for x in xs) for k in ("price", "solar")},
                )
                win = {}
                for variable in ("price", "solar"):
                    ws = [r[cohort + "_window"]["summary"][variable] for r in rows]
                    win[variable] = {
                        k: sum(x[k] for x in ws)
                        for k in ("n", "within", "done_by_end", "with_later_charge")
                    }
                    win[variable]["mean_in_block_energy_share"] = float(
                        np.mean([x["fraction_in_block"] for x in ws])
                    )
                groups[cohort + "_windows"] = win
            groups["curves"] = {}
            for h in ("8", "16"):
                xs = [r["original"][h] for r in rows]
                totals = {
                    k: sum(x[k] for x in xs)
                    for k in (
                        "n",
                        "within",
                        "applied",
                        "requested",
                        "cost",
                        "equal_energy_immediate_cost",
                        "solar_kwh",
                        "equal_energy_immediate_solar",
                    )
                }
                totals.update(
                    cost_reduction_pct=100
                    * (1 - totals["cost"] / totals["equal_energy_immediate_cost"]),
                    worst_shortfall=max(x["worst_shortfall"] for x in xs),
                    worst_excess=max(x["worst_excess"] for x in xs),
                )
                groups["curves"][h] = totals
            groups["dose_075"] = [r["dose"]["example_075"] for r in rows]
            groups["dose_passed"] = sum(r["dose"]["within"] for r in rows)
            groups["stop_passed"] = sum(r["stop"]["within"] for r in rows)
            groups["max_charge_on_arrival_at_target"] = max(
                r["stop"]["max_delivered"] for r in rows
            )
            if phase != "prior":
                folder = "final" if phase == "candidate" else "session"
                groups["export_sha256"] = {
                    s: hashlib.sha256(
                        (OUT / "models" / folder / f"v2-{w}-{s}.pth").read_bytes()
                    ).hexdigest()
                    for s in seeds
                }
            result[f"{phase}-{w}"] = groups
    return dict(
        protocol="Extension after the original short-window test. Mode retains full forecasts. One arm trains power from observed need, net solar and legal bounds. After 700 one-step reward updates, random low-cost blocks are used for own charge/idle continuations. The context arm restores the other observed power inputs after 2048 local updates and measures the return difference of nearby continuous powers. The mixed-window arm retains all original observed inputs throughout and alternates standard and variable-block training episodes, with lower exploration. The optional, separate contextual bandit chooses between two frozen self-trained EV readouts at plug-in, on their own operating returns over 196608 further sessions. No action labels; one shared encoder; mode and continuous power remain closed-loop throughout the stay. Reward unchanged. Training dates reused only within January–August. Across the original two seeds, model selection maximises the weaker development score: 65% worst and 35% mean across timing, service, dose, stop, original curves and completion by the end of a favourable block. Second untouched set of 32 distinct dates and new requests/times. Third frozen-seed replication. All timing pairs must satisfy service for both moved windows. Equal delivered energy for immediate-cost baseline. Each date has two possible block locations; block end leaves at least one hour before departure.",
        selection=selection,
        by_width=result,
    )


def final_dose_comparison(reports):
    from .dose_verify import blind, challenge, window_residuals

    torch.set_num_threads(1)
    independent_windows = challenge(blind(), 610970)
    rows = {}
    for w in WIDTHS:
        for arm in ("candidate", "calibrated", "joint"):
            seeds = {
                s: json.loads((reports / f"holdout-{arm}-{w}-{s}.json").read_text())
                for s in (*SEEDS, 109)
            }
            trials = list(seeds.values())
            short = [v["short"] for v in trials]
            group = dict(
                dates=len(trials[0]["days"]),
                sessions=sum(x["sessions"] for x in short),
                within=sum(x["near_target"] for x in short),
                mae=float(np.mean([x["mae_kwh"] for x in short])),
                worst_shortfall=max(x["worst_shortfall"] for x in short),
                worst_excess=max(x["worst_excess"] for x in short),
                strict_reached=sum(x["target_reached"] for x in short),
                over_01=sum(x["over_by_over_0_1"] for x in short),
                short_01=sum(x["short_by_over_0_1"] for x in short),
                stop=sum(v["stop"]["within"] for v in trials),
                max_charge_at_target=max(v["stop"]["max_delivered"] for v in trials),
                dose_pass=sum(v["dose"]["within"] for v in trials),
                dose_075=[v["dose"]["example_075"] for v in trials],
                trials=seeds,
            )
            for k in ("price", "solar"):
                long = [t["long"]["summary"][k] for t in trials]
                group[k] = dict(
                    pairs=sum(x[k]["pairs"] for x in short),
                    timing_and_target=sum(x[k]["timing_and_target"] for x in short),
                    early_complete=sum(x["done_by_end"] for x in long),
                    long_total=sum(x["n"] for x in long),
                    long_within=sum(x["within"] for x in long),
                    mean_in_block_share=float(np.mean([x["fraction_in_block"] for x in long])),
                )
            group["original"] = {}
            for hour in ("8", "16"):
                times = [t["original"][hour] for t in trials]
                sums = {
                    k: sum(l[k] for l in times)
                    for k in (
                        "n",
                        "within",
                        "applied",
                        "cost",
                        "equal_energy_immediate_cost",
                        "solar_kwh",
                        "equal_energy_immediate_solar",
                    )
                }
                sums.update(
                    cost_reduction_pct=100
                    * (1 - sums["cost"] / sums["equal_energy_immediate_cost"]),
                    worst_shortfall=max(l["worst_shortfall"] for l in times),
                    worst_excess=max(l["worst_excess"] for l in times),
                )
                group["original"][hour] = sums
            if arm == "joint":
                partial = [
                    window_residuals(
                        load_ev_policy(OUT / "models" / "validated" / f"v2-{w}-{s}.pth"),
                        independent_windows,
                    )
                    for s in (*SEEDS, 109)
                ]
                group["window_errors"] = {
                    v: dict(
                        short_over_01=sum(x[v]["short_over_01"] for x in partial),
                        excess_over_01=sum(x[v]["excess_over_01"] for x in partial),
                        max_short=max(x[v]["max_short"] for x in partial),
                        max_excess=max(x[v]["max_excess"] for x in partial),
                    )
                    for v in ("price", "solar")
                }
            rows[f"{arm}-{w}"] = group
    return dict(
        protocol="Final separately held-out September–December dates, excluding the earlier 20+32+32 dates. New fractional amounts and departure times, including requests between 1.01 and 1.34 kWh in the long-window challenge. Freeze the switch and upper-range continuous readout of the mixed-window V2. The local power response interpolates freely learned knot values as a function of observed remaining energy. No bound at target. One-quarter and multi-quarter physical returns, no labels. An additional arm holds the continuous response fixed and adapts the binary switch to its applied power. Switch duration is selected on development before reading the third test. Dates and energy requests from the earlier diagnostic are included only under explicit regressions. The interval used for calibration in the single-quarter phase stays within the physically protectable range.",
        selection=json.loads((OUT / "calibration-choice.json").read_text()),
        final_switch_selection=json.loads((OUT / "final-choice.json").read_text()),
        by_width=rows,
    )


def compile_report(reports):
    selection = json.loads((OUT / "selection.json").read_text())
    details = {}
    aggregate = {}
    for name, widths in [
        ("previous", (32, 64)),
        ("prefix", (32, 64)),
        ("native", WIDTHS),
        ("selected", WIDTHS),
    ]:
        for w in widths:
            vals = {}
            for seed in SEEDS:
                if name == "previous":
                    tag = f"old-{w}-{seed}"
                else:
                    m, step = (
                        (name, 12288)
                        if name != "selected"
                        else (selection[str(w)]["selected"][k] for k in ("method", "step"))
                    )
                    tag = f"{m}-{w}-{seed}-{step}"
                vals[seed] = json.loads((reports / f"test-{tag}.json").read_text())
            details[f"{name}-{w}"] = vals
            aggregate[f"{name}-{w}"] = combine(list(vals.values()))
    for w in WIDTHS:
        equal = [
            json.loads((reports / f"test-polish-{w}-{s}-2048.json").read_text()) for s in SEEDS
        ]
        aggregate[f"equal_budget-{w}"] = combine(equal)
        sel = selection[str(w)]["selected"]
        third = json.loads(
            (reports / f"test-{sel['method']}-{w}-109-{sel['step']}.json").read_text()
        )
        details[f"third-{w}"] = {109: third}
        aggregate[f"third-{w}"] = combine([third])
    energy_comparison = {
        f"{w}-{s}": json.loads((reports / f"cost-match-{w}-{s}.json").read_text())
        for w in WIDTHS
        for s in (*SEEDS, 109)
    }
    curve = json.loads((reports / "duration.json").read_text())
    verification = {}
    for w in WIDTHS:
        tag = selection[str(w)]["selected"]
        for seed in SEEDS:
            state = curve[f"{tag['method']}-{w}-{seed}-{tag['step']}"]
            native_replay = details[f"selected-{w}"][seed]["fresh"]
            difference = abs(native_replay["mae_kwh"] - state["mae"])
            assert difference < 2e-6 and native_replay["near_target"] == state["within"]
            assert native_replay["price"]["timing_and_target"] == state["price"]
            assert native_replay["solar"]["timing_and_target"] == state["solar"]
            verification[f"{w}-{seed}"] = difference
    native_dev = {
        p.stem[4:]: json.loads(p.read_text()) for p in sorted(reports.glob("dev-native-*.json"))
    }
    report = dict(
        protocol=dict(
            dataset="Core-30 Normal, 2023, Member_001; controlled isolated 20 kWh EV, 1.4–4 kW, quarter hours, efficiency 1",
            fitted_dates="January–August, development dates and validation fortnights excluded",
            seeds=SEEDS,
            independent_replication_seed=109,
            equal_budget="Fixed 49152 initial whole sessions plus 4096+2048 switch-refinement updates (96 starting states each) for all three widths. Two selection seeds. Independent seed uses the frozen per-width choice.",
            blind="32 new September–December dates; each date has price and solar pairs and two original measured curves. Repeat under two training seeds.",
            historical="Also repeat 20 earlier test dates, and 19 original night/day traces with earlier fixed requests.",
            target="Absolute departure error <= 0.1 kWh. Paired timing requires >0.1 kWh shifted into each of the two moved windows and both departures within tolerance.",
            dose="11 requests from 0.35 to 1 kWh in a final 15-min interval, under cheap grid, expensive grid and 4 kW solar.",
            previous="Newly evaluated saved matched-trajectory models with 3072 whole sessions, original rounded training requests.",
            native="Full V2 actor with four matched reward-only attempts; 3072, 6144, 12288 episodes with fractional requests.",
            prefix="Original V2 prefix PPO, both smaller widths, same 12288 fractional training episodes as native.",
            selected="Native V2 isolated-EV architecture with neutral backbone EV readouts and fixed continuous-interior band. Learn the EV mode and watts heads only, from actual costs, service and incremental overfill. Differentiable single-port physics; binary-switch refinement compares charge/idle followed by its own policy. No demonstrations, action labels or target cap.",
            training_budget="96 examples per update; initial gradient stage up to 393216 sessions. Refinements start from a common 49152-session checkpoint. Fork/polish batches have 96 sampled starting sessions with two counterfactual continuations each plus optional roll-in. Budgets are not interchangeable with full native episodes.",
            model_selection="Per width select one variant/stage for both seeds. Maximise weaker development seed (mean tiebreaker). 75% controlled composite, 25% original development curves.",
            forecasts="Retrospective perfect local outlooks; no 2024 training or evaluation. Original-curve immediate comparisons use same requested energy, marginal grid bill after household solar.",
            reward="Existing operating coefficients, +0.05 EUR per newly delivered kWh above the request; gamma=0.99 per hour. All weights random before training.",
            aggregation="Counts pool two seeded replays of the same held-out inputs. Shares are means of the two per-seed shares.",
        ),
        aggregate=aggregate,
        per_seed=details,
        confirmed_service_readout=confirmation(reports),
        final_dose=final_dose_comparison(reports),
        diagnosed_long_window_requests=json.loads((reports / "by-request.json").read_text()),
        equal_delivered_energy_cost=energy_comparison,
        development=selection,
        native_development=native_dev,
        fixed_learning_curves=curve,
        native_vs_batched_final_mae_difference=verification,
        export_check=json.loads((OUT / "reload.json").read_text()),
    )
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", action="store_true")
    p.add_argument("--learning-curves", type=Path)
    p.add_argument("--reports", type=Path)
    p.add_argument("--export-third", action="store_true")
    p.add_argument("--equal-energy", nargs=3, metavar=("WIDTH", "SEED", "OUTPUT"))
    args = p.parse_args()
    if args.learning_curves:
        learning_curves(args.learning_curves)
    if args.export:
        export()
    if args.export_third:
        export(third_only=True)
    if args.equal_energy:
        w, s, path = args.equal_energy
        equal_energy(int(w), int(s), Path(path))
    if args.reports:
        compile_report(args.reports)
