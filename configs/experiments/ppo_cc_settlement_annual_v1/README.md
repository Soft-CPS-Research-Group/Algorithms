# PPO/CC annual settlement protocol v1

Runnable canonical templates for the first paired annual comparison:

| File | Purpose | Learning |
|---|---|---|
| `smart_settlement_annual.yaml` | Neutral SMART reference | none |
| `cc_smart_settlement_annual_seed123.yaml` | CC over the exact SMART leaf | CC only |
| `ppo_settlement_annual_seed789.yaml` | Neutral replay of promoted PPO seed 789 | none |
| `cc_ppo_settlement_annual_seed789.yaml` | CC over promoted PPO seed 789 | CC only |

All four use the same dataset, full annual window, settlement contract and
final-only export policy. The two neutral/learned pairs contain byte-equivalent
leaf configuration blocks. Learned coordinators use the full `0.5--1.5`
multiplier range.

The PPO templates load the compact tracked checkpoint pack at
`artifacts/frozen_ppo/annual_v1/seed789`. It contains actor/value weights only;
optimizer, replay and exploration state are deliberately absent because the
leaf is frozen and deterministic.

Generate the canonical files again with:

```bash
.venv/bin/python scripts/generate_ppo_cc_settlement_templates.py
```

Validate the frozen contract with:

```bash
.venv/bin/pytest -q tests/test_ppo_cc_settlement_protocol.py
```

Seeds 123 and 456 for the PPO rows, and seeds 456 and 789 for CC-SMART, are
the later robustness expansion. They must use compact checkpoint packs and
configs generated from this same contract; they must not mutate these initial
paired templates in place.
