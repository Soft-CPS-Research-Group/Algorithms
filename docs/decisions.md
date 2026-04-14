# Design Decisions

## Purpose

This document captures key design decisions made during development. Each entry should be concise yet complete, explaining:
- **Problem**: What issue or question needed resolution
- **Decision**: The approach chosen
- **Rationale**: Why this approach was selected over alternatives

Keep entries brief but informative. Focus on decisions that affect architecture, data structures, or system behavior.

---

## Decision Log

### D001: Device ID Naming Consistency Validation

**Date**: 2026-04-15  
**Component**: `ObservationEnricher` (`algorithms/utils/observation_enricher.py`)

**Problem**: CityLearn currently uses single-instance batteries (`electrical_storage`) and multi-instance EV chargers (`electric_vehicle_storage_charger_1_1`). Future datasets may include multi-battery setups. Should we validate that device ID naming is consistent within each Controllable Asset (CA) type?

**Decision**: Added validation in `_extract_device_ids()` to enforce that each CA type uses consistent naming:
- **Single-instance**: All actions without suffix (e.g., `["electrical_storage"]`)
- **Multi-instance**: All actions with `_<device_id>` suffix (e.g., `["electrical_storage_1", "electrical_storage_2"]`)
- **Mixed (invalid)**: Rejects combinations like `["electrical_storage", "electrical_storage_1"]` with clear error message

**Rationale**:
1. **Forward compatibility**: When CityLearn adds multi-battery support, consistent naming will work automatically
2. **Early error detection**: Catches misconfigured datasets before they cause subtle bugs downstream
3. **Clear semantics**: Makes the single/multi-instance distinction explicit and unambiguous
4. **Matches CityLearn conventions**: Aligns with how EV chargers already handle multi-instance scenarios

**Alternatives considered**:
- No validation (rely on correct configuration): Risky, silent failures possible
- Auto-normalize to multi-instance format: Would break single-instance assumptions elsewhere
