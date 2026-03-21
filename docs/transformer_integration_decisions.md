# Transformer-Based MADDPG Integration Decisions

Records implementation decisions, rationale, alternatives considered, and deviations from the original plan.

## Implementation Start

**Date:** 2026-03-21

**Branch:** `gj/transformer_maddpg_integration`

---

## Decision Log

### D1: Feature-to-Token Mapping

**Decision:** Follow the plan exactly - CA tokens for controllable assets, SRO tokens for shared read-only observations, NFC token for non-flexible consumption.

**Rationale:** As specified in the plan, this mapping provides clear semantic separation and enables the 1-to-1 CA-to-output correspondence.

---

### D2: Critic Architecture  

**Decision:** Use mean-pooled encoder output with MLP head, implemented via strategy pattern.

**Rationale:** Simplest approach that mirrors centralized critic from standard MADDPG. Strategy pattern allows future alternatives.

---

### D3: Separate Encoders

**Decision:** Actor and Critic have separate Transformer encoders (not shared).

**Rationale:** Different learning objectives (policy gradient vs TD), avoid gradient coupling that can destabilize training.

---

### D4: Maximum Cardinality

**Decision:** max_tokens = 128 (configurable via YAML).

**Rationale:** Sufficient for production scenarios, can be adjusted per deployment.

---

### D5: Token Ordering

**Decision:** No positional embeddings; CA tokens first, then SRO, then NFC. Output slices first N_ca positions.

**Rationale:** 1-to-1 mapping is structural - output[i] corresponds to CA[i] by construction.

---

### D6: Replay Buffer Strategy

**Decision:** Pad to max cardinality with attention masks (PaddedStorageStrategy).

**Rationale:** Simpler implementation, acceptable memory overhead with max_tokens=128.

---

## Deviations from Plan

*None yet - implementation in progress.*

---

## Open Questions

*To be documented as they arise during implementation.*
