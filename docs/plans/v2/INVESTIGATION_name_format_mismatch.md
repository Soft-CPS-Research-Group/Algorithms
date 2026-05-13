# Investigation Report — Suspected Name-Format Mismatch (WP01–WP05)

**Status:** Read-only investigation. **Outcome: NO design/implementation mismatch.** The
initially suspected mismatch was a misreading of the bundled sample JSON file.

---

## TL;DR

The suspicion was: "WP01–WP05 layout builder/tokenizer expect `::`-delimited canonical
names (`charger::Building_1/charger_1_1::connected_state`, `district__month`), but live
`env.observation_names` and `EntityContractAdapter` output flat snake_case names
(`charging_phase_one_hot_charger_15_1_L1`, `outdoor_dry_bulb_temperature`)."

**This is wrong.** The flat snake_case names appear in the bundled file
`datasets/tmp_entity_obs_full_step2200_named.json` because that file is the
**simulator → wrapper input payload** (raw `tables.<entity>.features` per spec §8.1),
not adapter output. The `EntityContractAdapter.to_agent_observations(...)` method
*transforms* this raw payload into the canonical `::` / `district__` form
that the layout builder consumes. Pipeline is internally consistent.

---

## Q1. What is `EntityContractAdapter.to_agent_observations` actually emitting?

`utils/entity_adapter.py:200–321` — names are emitted with the canonical prefixes
the WP02/WP03 layout builder expects:

| Family | Format | Source line |
|--------|--------|-------------|
| District (singleton) | `district__<feature>` | `entity_adapter.py:215` |
| Building (singleton, unprefixed) | `<feature>` | `entity_adapter.py:224` |
| Per-storage | `storage::<storage_id>::<feature>` | `entity_adapter.py:239` |
| Per-PV | `pv::<pv_id>::<feature>` | `entity_adapter.py:250` |
| Per-charger | `charger::<charger_id>::<feature>` | `entity_adapter.py:261` |
| Connected EV ctx | `charger::<charger_id>::connected_ev::<feature>` | `entity_adapter.py:283, 291` |
| Incoming EV ctx | `charger::<charger_id>::incoming_ev::<feature>` | `entity_adapter.py:283, 294` |
| Operational counters | `active_(chargers\|storages\|pvs)_count` | `entity_adapter.py:297–299` |
| RBC aliases | `electric_vehicle_charger_state`, `electric_vehicle_soc`, … | `entity_adapter.py:304–319` |
| Flexibility flag | `electric_vehicle_is_flexible` | `entity_adapter.py:321` |

This matches the regexes in `algorithms/utils/entity_token_layout.py`
(`_PER_ASSET_LABELLED_RE = ^(storage|charger|pv)::([^:]+)::(.+)$`,
`_PER_ASSET_UNLABELLED_RE = ...`) **and** the `feature_patterns` in
`configs/tokenizers/entity_default.json` (e.g. `^district__month$`,
`^electrical_storage_soc$`). No translation layer is missing.

## Q2. What is the bundled sample JSON, then?

`datasets/tmp_entity_obs_full_step2200_named.json` is the **simulator-side entity
payload** — a single step captured from `softcpsrecsimulator==0.3.1`. Its top-level
schema is `{tables, edges, meta}` exactly as spec §8.1 prescribes for the *input*
to `EntityContractAdapter`:

```jsonc
{
  "schema": "/.../citylearn_three_phase_dynamic_topology_demo/schema.json",
  "tables": { "district": { "features": [...], "rows": [...] }, "building": {...}, ... },
  "edges":  { "building_to_charger": {...}, ... },
  "meta":   { "topology_version": ..., "entity_specs": ... }
}
```

The flat snake_case names like `outdoor_dry_bulb_temperature`,
`electrical_storage_soc`, and `charging_phase_one_hot_charger_15_1_L1` are the
**raw `tables.<entity>.features` arrays** — the per-table column labels the
adapter then prefixes/duplicates per asset. They are NOT what gets handed to the
agent.

## Q3. Why does `tests/_entity_sample_obs_names.py` exist?

It synthesizes the **adapter-output** observation_names list for `Building_1` from
the **adapter-input** sample payload, *without* needing to instantiate
`EntityContractAdapter` (which requires a CityLearn `env` for normalization
bounds). The file's docstring (lines 1–11) says exactly this:

> "This intentionally duplicates a small portion of adapter logic so the WP03
> layout-builder tests do not depend on the adapter being instantiable in a
> unit-test environment (the real adapter requires a CityLearn `env` instance
> and 2D numpy table payloads, while the on-disk sample stores tables as
> `{features, rows}`)."

The synthesis at `tests/_entity_sample_obs_names.py:46–137` mirrors
`utils/entity_adapter.py:213–329` step-by-step (district prefix → building
unprefixed → per-storage → per-PV → per-charger + connected_ev/incoming_ev →
counters → RBC aliases → flexibility flag). This is a test-only helper; it is
not "rewriting" anything that real code would do differently.

## Q4. Does `_DummyEntityEnv` in tests sidestep the issue?

Partially, but it confirms (does not contradict) the canonical-form contract.
`tests/test_entity_adapter.py:124–137` runs the *real* adapter on a real (but
synthetic) entity payload and asserts:

```python
assert observation_names[0][0] == "district__hour"
assert "electric_vehicle_charger_state" in observation_names[0]
assert any(name.startswith("charger::B1/C1::") for name in observation_names[0])
```

So the canonical form is exercised end-to-end against the live adapter, not
just synthesized in a fixture.

## Q5. Where does the wrapper hand names to the agent?

`utils/wrapper_citylearn.py`:
- `_apply_entity_layout(...)` at line 309 calls
  `self._entity_adapter.to_agent_observations(observation_payload)` (line 314)
  → `agent_observations, observation_names, observation_spaces`.
- Stores `self.observation_names = observation_names` (line 317).
- Calls `self.model.attach_environment(observation_names=self.observation_names, ...)`
  (line 372–373).

So the agent receives the canonical-form names directly from the adapter. The
wrapper does **not** apply any further canonicalization (and does not need to).

## Q6. Does `configs/tokenizers/entity_default.json` match adapter output?

Yes:

- District SROs use `^district__...$` patterns (lines 43–117), matching
  `f"district__{feature}"` at `entity_adapter.py:215`.
- Building singletons use unprefixed snake_case patterns (e.g.
  `^electrical_storage_soc$` line 124, `^charging_phase_one_hot_.+$` line 132,
  `^net_electricity_consumption$` line 153), matching the unprefixed building
  features at `entity_adapter.py:224`.
- Per-asset CA types use `adapter_prefix` keys: `"storage" / "charger"` (lines
  26–35) — consumed by the layout builder's `_PER_ASSET_*` regexes against the
  `storage::<id>::...` / `charger::<id>::...` form emitted at adapter lines
  239 / 261.
- Per-asset SRO types `pv` (`adapter_prefix: "pv::"`, line 180), `ev_connected`
  (`adapter_prefix: "charger::"`, `adapter_label: "connected_ev"`, lines
  183–188), `ev_incoming` (lines 190–195) — match adapter lines 250 / 283 with
  context_label.
- NFC `building_nfc` references unprefixed building features
  `non_shiftable_load - solar_generation` (lines 15–22) — match the building-block
  unprefixed names; `solar_generation` is added explicitly as an alias at
  `entity_adapter.py:327–329` if missing.
- Excluded features (lines 4–13) include `^electric_vehicle_charger_state$`
  and friends — exactly the RBC aliases the adapter emits at lines 304–319.

## Q7. Does the sample's schema affect anything?

The sample's `schema` path references
`citylearn_three_phase_dynamic_topology_demo` rather than the spec-mentioned
`_assets_only_demo`. This is irrelevant for the adapter / tokenizer contract —
both schemas produce entity payloads with the same top-level structure. The
sample is from the `_topology_demo` because it includes runtime topology
mutations (which the spec wants validated), and is the file the WP plans
reference for layout-builder tests.

## Q8. What is the actual test path that proves it works?

1. `tests/test_entity_adapter.py` runs the real `EntityContractAdapter` with a
   synthetic entity payload; asserts canonical `::` / `district__` names appear
   in the output (line 132–135).
2. `tests/_entity_sample_obs_names.py` synthesizes the same canonical names from
   the bundled sample (without instantiating the adapter, because the on-disk
   sample stores tables as `{features, rows}` rather than 2D numpy arrays).
3. `tests/test_entity_token_layout.py` feeds those synthesized canonical names
   into the WP03 `EntityTokenLayoutBuilder`; the regexes in
   `algorithms/utils/entity_token_layout.py` (and the `feature_patterns` in
   `configs/tokenizers/entity_default.json`) match correctly.
4. `tests/test_wrapper_entity_mode.py` mocks the env (`_DummyEntityEnv`) but
   the wrapper still calls the real adapter; the canonical names flow through
   `_apply_entity_layout` to `model.attach_environment`.

---

## Root Cause of the Confusion

The bundled JSON file (`datasets/tmp_entity_obs_full_step2200_named.json`) is
ambiguously named: "obs" suggests "observations the agent sees," but it is
actually the **raw simulator entity payload** that the adapter consumes. A
reader who opens the file and greps for names like `district__month` or
`storage::Building_1/...` will find none — and may incorrectly conclude that
the simulator/adapter emits flat snake_case to agents.

## Severity

**None.** No code change required. The pipeline is internally consistent and
all five WP02–WP04 test suites (where ported) exercise the canonical form.

## Recommendations

1. **Rename or document the sample file.** Either:
   - Rename `datasets/tmp_entity_obs_full_step2200_named.json` →
     `datasets/sample_entity_payload_step2200.json` (it's a simulator
     payload, not an "obs"); OR
   - Add a header comment / sidecar README in `datasets/` explaining that this
     file is the *input* to `EntityContractAdapter` (raw `{tables, edges,
     meta}` per spec §8.1), and that adapter output uses the canonical
     `district__` / `storage::<id>::` / `charger::<id>::` / `pv::<id>::` /
     `charger::<id>::connected_ev::` / `charger::<id>::incoming_ev::` prefixes
     (cite `utils/entity_adapter.py:213–329`).

2. **Cross-link the synthesis helper.** Add a one-line reference in
   `algorithms/utils/entity_token_layout.py` (near the regex definitions)
   pointing to `tests/_entity_sample_obs_names.py` and
   `utils/entity_adapter.py:213–329` so future readers see all three
   together.

3. **Strengthen the playbook (`docs/entity_interface_playbook_pt.md`)** to
   spell out: "the agent never sees raw `tables.<entity>.features` strings;
   `EntityContractAdapter` rewrites them into the canonical `::`/`district__`
   form that the layout builder and tokenizer config consume."

4. **(Optional) Add a regression test** that loads the bundled sample, runs it
   through the *real* adapter (constructed with a minimal stub env exposing
   `entity_specs` only), and asserts adapter output exactly matches
   `tests/_entity_sample_obs_names.load_sample_observation_names_for_first_building()`.
   This would catch any future drift between the synthesis helper and the
   real adapter.

No WP01–WP05 plan changes are required.
