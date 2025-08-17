from torch.nn import Module
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numpy.typing as npt

class BaseAgent(Module):
    def __init__(self):
        Module.__init__(self)  # Initialize nn.Module
        ABC.__init__(self) #Initializes base class
        # Add any shared logic or attributes here
    
    @abstractmethod
    def predict(self, observations: List[npt.NDArray[np.float64]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for the current time step."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update replay buffer and networks."""
        pass

    @abstractmethod
    def save_checkpoint(self, step: int) -> None:
        """Persist model state for later resumption."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self) -> None:
        """Restore model state from a previously saved checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def export_to_onnx(self, log_dir: str) -> None:
        """Export a representation of the agent for serving."""
        raise NotImplementedError

    

