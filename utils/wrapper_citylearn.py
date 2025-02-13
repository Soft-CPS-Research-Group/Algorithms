from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.base_agent import BaseAgent
from loguru import logger
import mlflow
import numpy.typing as npt
from utils.preprocessing import *

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
        self.checkpoint_interval = config["algorithm"]["hyperparameters"]["checkpoint_interval"]


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
        super().learn(episodes, deterministic, deterministic_finish, logging_level)
        self.model.save_models(mlflow.active_run().run.info.run_id)
        return

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

        #if self.time_step % self.checkpoint_interval == 0:
            #self.save_checkpoint(mlflow.active_run().run.info.run_ids, self.current_step)

        if args:
            # Process current observations (assumed to be the 1st argument)
            observations = args[0]
            encoded_observations = self.get_all_encoded_observations(observations)

            # Process next observations (assumed to be the 4th argument)
            if len(args) >= 4:
                next_observations = args[3]
                encoded_next_observations = self.get_all_encoded_observations(next_observations)
            else:
                encoded_next_observations = None  # or handle this case appropriately

            # Convert args to a list to update the tuple
            args_list = list(args)
            args_list[0] = encoded_observations
            if encoded_next_observations is not None:
                args_list[3] = encoded_next_observations
            args = tuple(args_list)

        # Pass updated parameters to model.update()
        return self.model.update(
            initial_exploration_done=self.initial_exploration_done,
            update_step=self.update_step,
            update_target_step=self.update_target_step, time_step=self.time_step,
            *args, **kwargs
        )

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array(
            [j for j in np.hstack(self.encoders[index] * np.array(observations, dtype=float)) if j is not None],
            dtype=float
        )

    def get_all_encoded_observations(self, observations: List[List[float]]) -> List[npt.NDArray[np.float64]]:
        return [
            np.array(
                [j for j in np.hstack(self.encoders[idx] * np.array(obs, dtype=float)) if j is not None],
                dtype=float
            )
            for idx, obs in enumerate(observations)
        ]

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

                elif any(item in n for item in ["connected_state", "incoming_state"]):
                    e.append(OnehotEncoding([0,1]))

                elif any(item in n for item in ["required_soc_departure", "estimated_soc_arrival", "electric_vehicle_soc"]):
                    e.append(NormalizeWithMissing(s.low[i], s.high[i]))

                elif any(item in n for item in ["departure_time", "arrival_time"]):
                    e.append(OnehotEncoding([-0.1] + list(range(0, 25)))) #-0.1 encodes missing values

                elif n in ['day_type']:
                    e.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))

                elif n in ["daylight_savings_status"]:
                    e.append(OnehotEncoding([0, 1]))

                elif n in remove_features:
                    e.append(RemoveFeature())

                else:
                    e.append(NoNormalization())

            encoders.append(e)

        logger.debug("Encoders SET")

        return encoders