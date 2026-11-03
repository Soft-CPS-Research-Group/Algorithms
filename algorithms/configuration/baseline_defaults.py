"""Lightweight policy defaults shared by validation and direct construction."""

EV_SERVICE_SOC_TOLERANCE = 0.001

# Fixed reference tariff: Core30 2023 OMIE terciles, rounded to 4 decimals.
# Selection uses prices only, never controller outcomes or 2024 data.
BASIC_PRICE_DEFAULTS = {
    "basic_price_policy": "fixed_thresholds",
    "basic_cheap_price_eur_kwh": 0.0765,
    "basic_expensive_price_eur_kwh": 0.1078,
}
COMMUNITY_IMPORT_THRESHOLD_KW = 15.0

NORMAL_DEFAULTS = {
    "storage_charge_rate": 1.0,
    "storage_discharge_rate": 1.0,
    "pv_surplus_threshold_kw": 0.05,
    "normal_storage_discharge_import_threshold_kw": 0.05,
    "ev_service_soc_tolerance": EV_SERVICE_SOC_TOLERANCE,
}

SMART_DEFAULTS = {"price_charge_rate": 0.0}
ECONOMIC_SMART_DEFAULTS = {"price_charge_rate": 0.60}


def policy_defaults(name):
    if name == "RBCBasicPolicy":
        return BASIC_PRICE_DEFAULTS
    if name == "RBCCommunityPolicy":
        return {
            **ECONOMIC_SMART_DEFAULTS,
            "community_import_threshold_kw": COMMUNITY_IMPORT_THRESHOLD_KW,
        }
    if name in {"Smart", "RBCSmartLocalPolicy"}:
        return ECONOMIC_SMART_DEFAULTS
    if name == "NormalTargetPolicy":
        return {**NORMAL_DEFAULTS, "ev_normal_target_soc": 0.0}
    if name in {"NormalPolicy", "NormalNoBatteryPolicy"}:
        return NORMAL_DEFAULTS
    if name in {
        "RBCSmartPolicy",
        "SignalAwareRBC",
        "SignalAwareRBCSmartLocal",
        "SignalAwareRBCCommunity",
        "SignalDirectedRBC",
    }:
        return SMART_DEFAULTS
    return {}
