"""Lightweight policy defaults shared by validation and direct construction."""

NORMAL_DEFAULTS = {
    "storage_charge_rate": 1.0,
    "storage_discharge_rate": 1.0,
    "pv_surplus_threshold_kw": 0.05,
    "normal_storage_discharge_import_threshold_kw": 0.05,
    "ev_service_soc_tolerance": 0.001,
}

SMART_DEFAULTS = {"price_charge_rate": 0.0}


def policy_defaults(name):
    if name in {"NormalPolicy", "NormalNoBatteryPolicy"}:
        return NORMAL_DEFAULTS
    if name in {
        "RBCSmartPolicy",
        "RBCSmartLocalPolicy",
        "RBCCommunityPolicy",
        "SignalAwareRBC",
        "SignalAwareRBCSmartLocal",
        "SignalAwareRBCCommunity",
        "SignalDirectedRBC",
    }:
        return SMART_DEFAULTS
    return {}
