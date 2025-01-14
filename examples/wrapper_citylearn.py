from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.base_agent import BaseAgent

class Wrapper_CityLearn(RLC):
    def __init__(self, env: CityLearnEnv, model: BaseAgent, **kwargs):
        """
        Wrapper for CityLearn RLC that delegates custom behavior to a BaseAgent model.

        Parameters:
        - env: CityLearnEnv instance for the simulation environment.
        - model: BaseAgent instance implementing custom predict and update logic.
        - **kwargs: Additional arguments passed to the RLC constructor.
        """
        super().__init__(env, **kwargs)
        self.model = model  # Custom model logic

    def learn(self, episodes=None, deterministic=None, deterministic_finish=None, logging_level=None):
        """
        Placeholder method for learn.
        
        Currently, this does nothing custom and uses the RLC implementation via super().
        """
        return super().learn(episodes, deterministic, deterministic_finish, logging_level)

    def predict(self, observations, deterministic=None):
        """
        Updates the predict action logic. It now uses a mix of algorithm and the next time step.
        """
        actions = self.model.predict(observations, deterministic)
        self.actions = actions
        self.next_time_step()
        return actions

    def update(self, *args, **kwargs):
        """
        Delegates the update logic to the Algorithm.
        """
        return self.model.update(*args, **kwargs)

