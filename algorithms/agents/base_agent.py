import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(self):
        super(BaseAgent, self).__init__()

    def act(self, state):
        """
        Compute the action given the state.
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        Update the agent's parameters.
        """
        raise NotImplementedError