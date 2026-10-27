"""Agent-owned environment checks called by the common preflight service."""


def validate_transformer_maddpg(stage, wrapper):
    dimensions = list(wrapper.action_dimension)
    if not dimensions or min(dimensions) <= 0 or len(set(dimensions)) != 1:
        raise ValueError("TransformerMADDPG requires uniform, nonzero action dimensions")


def validate_timarl(stage, wrapper):
    from algorithms.ti_marl.compiler.compiler import TypedInterfaceCompiler

    parameters = stage["hyperparameters"]
    compiler = TypedInterfaceCompiler(
        contract_version=parameters["contract_version"],
        typed_interfaces_dir=parameters["typed_interfaces_dir"],
        simulator_bindings_path=parameters.get("simulator_bindings_path"),
    )
    compiler.adapter.attach_entity_specs(
        wrapper.env.entity_specs,
        seconds_per_time_step=wrapper.env.seconds_per_time_step,
    )
