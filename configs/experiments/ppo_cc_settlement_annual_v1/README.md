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
leaf configuration blocks. Learned coordinators use the `0.5--1.3`
multiplier range used by Pedro's comparable annual CC-TD3 run.

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

Generate the ignored local end-to-end smoke configs with:

```bash
.venv/bin/python scripts/generate_ppo_cc_settlement_smokes.py
```

The smoke window contains 385 dataset rows/384 environment transitions. The
neutral rows use one deterministic pass. The learned CC rows use three passes:
one complete BC collection/pretraining pass, one trainable pass that forces a
complete 96-decision PPO update, and one final deterministic export pass. These
configs validate runtime mechanics only and are not annual performance
evidence.

Seeds 123 and 456 for the PPO rows, and seeds 456 and 789 for CC-SMART, are
the later robustness expansion. They must use compact checkpoint packs and
configs generated from this same contract; they must not mutate these initial
paired templates in place.
