"""Schema hashing and checkpoint-composition compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from algorithms.ti_marl.contracts.models import content_hash


@dataclass(frozen=True)
class CompatibilitySignature:
    contract_version: str
    agent_schema_hash: str
    type_registry_hash: str
    health_rules_hash: str
    compiler_hash: str
    supported_module_types: Tuple[str, ...]
    supported_action_group_types: Tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        contract_version: str,
        agent_schema: Mapping[str, Any],
        type_registry: Mapping[str, Any],
        health_rules: Mapping[str, Any],
        compiler_version: str,
    ) -> "CompatibilitySignature":
        return cls(
            contract_version=str(contract_version),
            agent_schema_hash=content_hash(agent_schema),
            type_registry_hash=content_hash(type_registry),
            health_rules_hash=content_hash(health_rules),
            compiler_hash=content_hash({"compiler_version": compiler_version}),
            supported_module_types=tuple(sorted(str(key) for key in type_registry.get("entity_types", {}))),
            supported_action_group_types=tuple(
                sorted(str(key) for key in type_registry.get("action_group_types", {}))
            ),
        )

    def accepts(self, other: "CompatibilitySignature") -> bool:
        """Accept composition changes, but never semantic-contract changes."""

        return (
            self.contract_version == other.contract_version
            and self.agent_schema_hash == other.agent_schema_hash
            and self.type_registry_hash == other.type_registry_hash
            and self.health_rules_hash == other.health_rules_hash
            and self.compiler_hash == other.compiler_hash
            and self.supported_module_types == other.supported_module_types
            and self.supported_action_group_types == other.supported_action_group_types
        )

    def accepts_explicit_compiler_migration(
        self,
        other: "CompatibilitySignature",
    ) -> bool:
        """Accept only a compiler-version change on an otherwise identical surface."""

        return (
            self.contract_version == other.contract_version
            and self.agent_schema_hash == other.agent_schema_hash
            and self.type_registry_hash == other.type_registry_hash
            and self.health_rules_hash == other.health_rules_hash
            and self.supported_module_types == other.supported_module_types
            and self.supported_action_group_types
            == other.supported_action_group_types
        )
