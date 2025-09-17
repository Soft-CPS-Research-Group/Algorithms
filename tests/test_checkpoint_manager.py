from pathlib import Path

from utils.checkpoint_manager import CheckpointManager


class DummyAgent:
    def __init__(self):
        self.saved_steps = []

    def save_checkpoint(self, output_dir: str, step: int):
        path = Path(output_dir) / f"dummy_{step}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint")
        self.saved_steps.append(step)
        return str(path)


def test_checkpoint_manager_saves(tmp_path):
    manager = CheckpointManager(base_dir=str(tmp_path), interval=5)
    agent = DummyAgent()

    # Should skip when step not aligned
    assert manager.maybe_save(agent, step=4, initial_exploration_done=True, update_step=True) is None

    # Should skip before exploration done
    assert manager.maybe_save(agent, step=5, initial_exploration_done=False, update_step=True) is None

    # Should save when conditions met
    path = manager.maybe_save(agent, step=5, initial_exploration_done=True, update_step=True)
    assert path is not None
    assert Path(path).exists()
    assert agent.saved_steps == [5]
