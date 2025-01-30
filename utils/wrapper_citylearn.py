from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.base_agent import BaseAgent
from utils.mlflow_helper import log_param_to_mlflow
from loguru import logger
import numpy as np
import numpy.typing as npt
from citylearn.preprocessing import *

class Wrapper_CityLearn(RLC):
    def __init__(self, env: CityLearnEnv, model: BaseAgent = None, config = None, **kwargs):
        """
        Wrapper for CityLearn RLC that delegates custom behavior to a BaseAgent model.

        Parameters:
        - env: CityLearnEnv instance for the simulation environment.
        - model: BaseAgent instance implementing custom predict and update logic.
        - **kwargs: Additional arguments passed to the RLC constructor.
        """
        super().__init__(env, **kwargs)
        self.model = model  # Custom model logic, it can be none upon initialization
        self.exploration_done = False
        self.initial_exploration_done = False
        self.update_step = False
        self.update_target_step = False
        self.steps_between_training_updates = config["algorithm"]["hyperparameters"]["steps_between_training_updates"]
        self.end_initial_exploration_time_step = config["algorithm"]["hyperparameters"]["end_initial_exploration_time_step"]
        self.end_exploration_time_step = config["algorithm"]["hyperparameters"]["end_exploration_time_step"]
        self.target_update_interval = config["algorithm"]["hyperparameters"]["target_update_interval"]


    def set_model(self, model: BaseAgent):
        """
        Set the model after initialization.
        """
        self.model = model

    def learn(self, episodes=None, deterministic=None, deterministic_finish=None, logging_level=None):
        """
        Placeholder method for learn.
        
        Currently, this does nothing custom and uses the RLC implementation via super().
        """
        if self.model is None:
            raise ValueError("Model is not set. Use `set_model` to provide a model.")
        return super().learn(episodes, deterministic, deterministic_finish, logging_level)

    def predict(self, observations, deterministic=None):
        """
        Updates the predict action logic. It now uses a mix of algorithm and the next time step.
        """
        if self.model is None:
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        # Determine exploration phase
        self.exploration_done = self.time_step >= self.end_exploration_time_step
        logger.debug("Exploration ended." if self.exploration_done else "Doing Exploration.")

        encoded_observations = self.get_all_encoded_observations(observations)
        actions = self.model.predict(encoded_observations, deterministic)
        self.actions = actions
        self.next_time_step()
        return actions

    def update(self, *args, **kwargs):
        """
        Delegates the update logic to the Algorithm, encoding observations before passing them.
        """
        if self.model is None:
            logger.error("Model is not set. Use `set_model` to provide a model.")
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        # Determine exploration phase
        self.initial_exploration_done = self.time_step >= self.end_initial_exploration_time_step
        logger.debug("Initial Exploration ended." if self.exploration_done else "Doing Initial Exploration.")

        # Determine whether to update
        self.update_step = self.time_step % self.steps_between_training_updates != 0
        logger.debug("Time step - Doing Update" if self.update_step else "Time step - Skipping Update")

        # Determine whether to update the target networks
        self.update_target_step = self.time_step % self.target_update_interval == 0
        logger.debug(
            "Time step - Doing Target Update" if self.update_target_step else "Time step - Skipping Target Update")

        # Extract observations from args
        if args:
            observations = args[0]  # First argument should be observations
            encoded_observations = self.get_all_encoded_observations(observations)
            args = (encoded_observations,) + args[1:]  # Replace original observations with encoded ones

        # Pass updated parameters to model.update()
        return self.model.update(
            initial_exploration_done=self.initial_exploration_done,
            update_step=self.update_step,
            update_target_step=self.update_target_step,
            *args, **kwargs
        )

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index] * np.array(observations, dtype=float)) if j != None],
                        dtype=float)

    def get_all_encoded_observations(self, observations: List[List[float]]) -> List[npt.NDArray[np.float64]]:


        logger.debug(observations)
        encoded_observations = []
        for index, obs in enumerate(observations):
            encoded_observations.append(
                np.array(
                    [j for j in np.hstack(self.encoders[index] * np.array(obs, dtype=float)) if j is not None],
                    dtype=float
                )
            )
        logger.debug(encoded_observations)
        return encoded_observations

    def set_encoders(self) -> List[List[Encoder]]:
        r"""Get observation value transformers/encoders for use in MARLISA agent internal regression model.

        The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
        `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
        and `Normalize` for observations with known minimum and maximum boundaries.

        Returns
        -------
        encoders : List[Encoder]
            Encoder classes for observations ordered with respect to `active_observations`.
        """

        encoders = []

        for o, s in zip(self.observation_names, self.observation_space):
            e = []

            remove_features = [
                'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
                'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h',
                'outdoor_relative_humidity', 'outdoor_relative_humidity_predicted_6h',
                'outdoor_relative_humidity_predicted_12h', 'outdoor_relative_humidity_predicted_24h',
                'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
                'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h'
            ]

            for i, n in enumerate(o):
                if n in ['month', 'hour']:
                    e.append(PeriodicNormalization(s.high[i]))

                elif any(item in n for item in ["required_soc_departure", "estimated_soc_arrival", "ev_soc"]):
                    e.append(Normalize(s.low[i], s.high[i]))

                elif any(item in n for item in ["estimated_departure_time", "estimated_arrival_time"]):
                    e.append(OnehotEncoding([-1] + list(range(0, 25))))

                elif n in ['day_type']:
                    e.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))

                elif n in ["daylight_savings_status"]:
                    e.append(OnehotEncoding([0, 1]))

                elif n in remove_features:
                    e.append(RemoveFeature())

                else:
                    e.append(NoNormalization())

            encoders.append(e)

        return encoders