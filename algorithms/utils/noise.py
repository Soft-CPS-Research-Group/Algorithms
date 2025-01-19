import numpy as np
from loguru import logger

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

def add_noise(action, sigma, bias=0.0, action_bounds=(-1, 1)):
    """Generate and add exploration noise to an action.

    Parameters
    ----------
    action : np.ndarray
        Deterministic action to which noise will be added.
    sigma : float
        Standard deviation for the noise.
    bias : float, optional
        Bias to subtract from the generated noise, by default 0.0.
    action_bounds : tuple, optional
        Bounds for clipping actions, by default (-1, 1).

    Returns
    -------
    np.ndarray
        Action with added noise, clipped to valid range.
    """
    logger.debug("Adding noise to action in helper.")
    noise = np.random.normal(scale=sigma, size=action.shape)
    biased_noise = noise - bias
    noisy_action = np.clip(action + biased_noise, *action_bounds)

    logger.debug(f"Action: {action}, Noise: {noise}, Biased Noise: {biased_noise}, Noisy Action: {noisy_action}")
    return noisy_action
