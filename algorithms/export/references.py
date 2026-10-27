"""Bounded observations/outputs of the actual final policy, captured before export."""

from copy import deepcopy

from algorithms.export.portable import encode


def set_capture_step(unit, step):
    if not hasattr(unit, "_export_references"):
        return
    unit._export_episode_step = step
    for name in ("stages", "agents"):
        for child in getattr(unit, name, []):
            set_capture_step(child, step)


def begin_capture(unit, config):
    unit._export_reference_limit = int((config.get("bundle") or {}).get("verification_samples", 32))
    unit._export_references = []
    for key in ("_export_initial_state", "_export_state_error"):
        if hasattr(unit, key):
            delattr(unit, key)
    for name in ("stages", "agents"):
        for child in getattr(unit, name, []):
            begin_capture(child, config)


def capture_input(unit, observations, context):
    samples = getattr(unit, "_export_references", None)
    if samples is None or len(samples) >= unit._export_reference_limit:
        return None
    if (
        not samples
        and getattr(getattr(unit, "capabilities", None), "learning", None) == "none"
        and not hasattr(unit, "agents")
    ):
        from algorithms.export.policy_json import policy_state

        try:
            unit._export_initial_state = policy_state(unit)
        except TypeError as exc:
            unit._export_state_error = str(exc)
    return {
        "observations": encode(deepcopy(observations)),
        "context": encode(deepcopy(context)),
        "raw_observations": encode(deepcopy(getattr(unit, "_raw_observations", None))),
        "encoded_observations": encode(deepcopy(getattr(unit, "_encoded_observations", None))),
        "episode_step": getattr(unit, "_export_episode_step", None),
    }


def capture_output(unit, sample, output):
    if sample is not None:
        sample["output"] = encode(deepcopy(output))
        unit._export_references.append(sample)


def capture_submission(unit, actions, step):
    samples = getattr(unit, "_export_references", [])
    if samples and samples[-1]["episode_step"] == step:
        samples[-1]["submitted_actions"] = encode(deepcopy(actions))
