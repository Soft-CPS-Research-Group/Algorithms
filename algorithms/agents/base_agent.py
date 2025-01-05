from torch.nn import Module
from abc import ABC, abstractmethod

class BaseAgent(Module):
    def __init__(self):
        Module.__init__(self)  # Initialize nn.Module
        ABC.__init__(self) #Initializes base class
        # Add any shared logic or attributes here
    
    @abstractmethod
    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for the current time step."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update replay buffer and networks."""
        pass

    

