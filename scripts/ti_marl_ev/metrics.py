"""Paired timing, energy/service, and after-target decisions for the EV experiments."""

import statistics


def measures(rows):
    service = []
    extra = []
    bill = []
    misses = []
    minimum = []
    late_charge = 0
    after_target = 0
    pairs = {k: [] for k in ("price", "solar")}
    for row in rows:
        s = row["session"]
        n = s["quarters"]
        mid = n // 2
        early = row["early"]
        late = row["late"]
        for run in (early, late):
            total = sum(run["kwh"])
            error = total - s["request_kwh"]
            assert abs(error - (run["extra"] - run["miss"])) < 1e-7
            sofar = 0.0
            for q in run["kwh"]:
                assert -1e-7 <= q <= 1.0 + 1e-7
                if sofar >= s["request_kwh"] - 1e-7:
                    after_target += 1
                    late_charge += int(q > 1e-6)
                sofar += q
            service.append(abs(error))
            extra.append(run["extra"])
            misses.append(run["miss"])
            minimum.append(run["minimum_miss"])
            bill.append(run["bill"])
        a = sum(early["kwh"][:mid]) - sum(late["kwh"][:mid])
        b = sum(late["kwh"][mid:]) - sum(early["kwh"][mid:])
        assert abs(row["shift"] - a) < 1e-8
        near = max(early["extra"], early["miss"], late["extra"], late["miss"]) <= 0.1 + 1e-8
        pairs[s["variable"]].append((min(a, b), near, a))
    summary = {
        "sessions": len(service),
        "mae_kwh": statistics.mean(service),
        "mean_extra_kwh": statistics.mean(extra),
        "mean_missing_kwh": statistics.mean(misses),
        "bill_eur": statistics.mean(bill),
        "near_target": sum(x <= 0.1 + 1e-8 for x in service),
        "target_reached": sum(x <= 0.001 for x in misses),
        "protected": sum(x <= 0.001 for x in minimum),
        "charged_after_target": late_charge,
        "decisions_after_target": after_target,
    }
    for name, ls in pairs.items():
        if ls:
            summary[name] = {
                "pairs": len(ls),
                "timing": sum(a > 0.1 for a, _, _ in ls),
                "timing_and_target": sum(a > 0.1 and near for a, near, _ in ls),
                "timing_legacy": sum(a > 0.1 for _, _, a in ls),
                "mean_shift_kwh": statistics.mean(a for a, _, _ in ls),
            }
    return summary
