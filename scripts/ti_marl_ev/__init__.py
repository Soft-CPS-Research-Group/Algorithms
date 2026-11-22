"""Pretrain and test the isolated EV decisions of an existing V2 research actor.

From the Algorithms checkout, supply the existing V2 checkpoint and probe
inputs via the paths in protocol.py. A small-network example:

    python -m scripts.ti_marl_ev.demonstrations 32 short.npy
    python -m scripts.ti_marl_ev.fit 32 43 boundary short.npy
    python -m scripts.ti_marl_ev.demonstrations 32 night.npy --wide --count 9000
    python -m scripts.ti_marl_ev.fit 32 43 wide short.npy night.npy
    python -m scripts.ti_marl_ev.rl 32 43 wide 0.01 bc

The last command runs ordinary PPO with no expert. Use 'cold' for the matched
control. Select the actor learning-rate multiplier on development sessions.
"""
