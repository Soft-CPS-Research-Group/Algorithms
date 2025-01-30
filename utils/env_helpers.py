from typing import Dict, List
import numpy as np
from citylearn.citylearn import CityLearnEnv

def extract_environment_dimensions(env: CityLearnEnv) -> Dict[str, any]:
    """
    Extract observation and action space dimensions from the environment.

    Parameters:
    - env: CityLearnEnv instance.

    Returns:
    - A dictionary with relevant dimensions.
    """
    observation_dimensions_agents = [space.shape[0] for space in env.observation_space]
    action_dimensions_agents = [space.shape[0] for space in env.action_space]

    return {
        "observation_dimensions_REC": sum(observation_dimensions_agents),
        "observation_dimensions_agents": observation_dimensions_agents,
        "action_dimensions_REC": sum(action_dimensions_agents),
        "action_dimensions_agents": action_dimensions_agents,
    }



