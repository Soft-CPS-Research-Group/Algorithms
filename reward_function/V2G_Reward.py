from typing import Any, List, Mapping, Union
from citylearn.reward_function import RewardFunction

from typing import Any, List, Mapping, Union
import numpy as np
from citylearn.reward_function import RewardFunction

class V2GPenaltyReward(RewardFunction):
    """Rewards with considerations for car charging and for MADDPG Mixed environments."""

    def __init__(self, env_metadata: Mapping[str, Any],
                 peak_percentage_threshold=0.10,
                 ramping_percentage_threshold=0.10,
                 peak_penalty_weight=20,
                 ramping_penalty_weight=15,
                 energy_transfer_bonus=10,
                 window_size=6,
                 penalty_no_car_charging=-5,
                 penalty_battery_limits=-2,
                 penalty_soc_under_5_10=-5,
                 reward_close_soc=10,
                 reward_self_ev_consumption=5,
                 community_weight=0.2,
                 reward_extra_self_production=5,
                 squash=0):
        super().__init__(env_metadata)
        self.rolling_window = []

        # Setting the parameters
        self.PEAK_PERCENTAGE_THRESHOLD = peak_percentage_threshold
        self.RAMPING_PERCENTAGE_THRESHOLD = ramping_percentage_threshold
        self.PEAK_PENALTY_WEIGHT = peak_penalty_weight
        self.RAMPING_PENALTY_WEIGHT = ramping_penalty_weight
        self.ENERGY_TRANSFER_BONUS = energy_transfer_bonus
        self.WINDOW_SIZE = window_size
        self.PENALTY_NO_CAR_CHARGING = penalty_no_car_charging
        self.PENALTY_BATTERY_LIMITS = penalty_battery_limits
        self.PENALTY_SOC_UNDER_5_10 = penalty_soc_under_5_10
        self.REWARD_CLOSE_SOC = reward_close_soc
        self.COMMUNITY_WEIGHT = community_weight
        self.SQUASH = squash
        self.REWARD_EXTRA_SELF_PRODUCTION = reward_extra_self_production
        self.REWARD_SELF_EV_CONSUMPTION = reward_self_ev_consumption

    def calculate_building_reward(self, observation: Mapping[str, Union[int, float]]) -> float:
        """Calculate individual building reward."""
        net_energy = observation['net_electricity_consumption']
        reward_type = observation.get('reward_type', None)
        reward = 0

        if reward_type == "C":  # Pricing-based reward
            price = observation.get('electricity_pricing', 0.0)
            if net_energy > 0:  # Consuming from the grid
                reward = -price * net_energy
            else:  # Exporting to the grid
                reward = 0.80 * price * abs(net_energy)
        elif reward_type == "G":  # Reducing carbon emissions
            carbon_intensity = observation.get('carbon_intensity', 0.0)
            reward = carbon_intensity * (net_energy * -1)
        elif reward_type == "Z":  # Increasing zero net energy
            if net_energy > 0:
                reward = -net_energy
            else:
                reward = abs(net_energy) * 0.5
        else:
            reward = net_energy * -1

        # EV penalties or rewards
        reward += self.calculate_ev_penalty(observation, reward, net_energy)

        return reward

    def calculate_ev_penalty(self, observation: Mapping[str, Union[int, float]], current_reward: float, net_energy: float) -> float:
        """Calculate penalties based on EV specific logic."""
        penalty = 0
        penalty_multiplier = abs(current_reward)

        chargers = observation.get('chargers', [])
        for c in chargers:
            last_connected_car = c.get('last_connected_car', None)
            last_charged_value = c.get('last_charging_action_value', 0.0)

            # 1. Penalty for charging when no car is present
            if last_connected_car is None and last_charged_value != 0:
                penalty += self.PENALTY_NO_CAR_CHARGING * penalty_multiplier

            if last_connected_car is not None:
                soc = last_connected_car.get('soc', 0.0)
                capacity = last_connected_car.get('capacity', 1.0)
                required_soc = last_connected_car.get('required_soc_departure', 0.0)

                # 3. Penalty for exceeding the battery's limits
                if soc + last_charged_value > capacity or soc + last_charged_value < 0:
                    penalty += self.PENALTY_BATTERY_LIMITS * penalty_multiplier

                # 4. Penalty or reward for SoC differences
                soc_diff = (soc / capacity) - required_soc
                penalty += self.REWARD_CLOSE_SOC * (1 - abs(soc_diff))

        return penalty

    def calculate_community_reward(self, observations: List[Mapping[str, Union[int, float]]], rewards: List[float]) -> List[float]:
        """Calculate community building reward."""
        community_net_energy = sum(o['net_electricity_consumption'] for o in observations)

        if len(self.rolling_window) >= self.WINDOW_SIZE:
            self.rolling_window.pop(0)
        self.rolling_window.append(community_net_energy)

        average_past_consumption = sum(self.rolling_window) / len(self.rolling_window)
        dynamic_peak_threshold = average_past_consumption * (1 + self.PEAK_PERCENTAGE_THRESHOLD)
        ramping = community_net_energy - average_past_consumption

        community_reward = 0
        if community_net_energy > dynamic_peak_threshold:
            community_reward -= (community_net_energy - dynamic_peak_threshold) * self.PEAK_PENALTY_WEIGHT
        if abs(ramping) > dynamic_peak_threshold:
            community_reward -= abs(ramping) * self.RAMPING_PENALTY_WEIGHT

        community_reward += sum(-o['net_electricity_consumption'] * self.ENERGY_TRANSFER_BONUS for o in observations if o['net_electricity_consumption'] < 0)

        updated_rewards = [r + community_reward * self.COMMUNITY_WEIGHT for r in rewards]

        return updated_rewards

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        raw_reward_list = [self.calculate_building_reward(o) for o in observations]
        reward_list = self.calculate_community_reward(observations, raw_reward_list)

        if self.SQUASH:
            reward_list = [np.tanh(r) for r in reward_list]

        if self.central_agent:
            return [sum(reward_list)]
        return reward_list