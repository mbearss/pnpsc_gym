import gym
import numpy as np


class RateAdjWrapper(gym.Wrapper):
    """
    Wrapper to the PNPSC environment to use actions as adjustments to rates rather than final rate values
    """

    def __init__(self, env):
        """
        Create a wrapper for the PNPSC environment
        :param env: PNPSC environment to wrap
        """
        super().__init__(env)
        self.env = env
        self.last_rates = None

        # player can attempt to shift the rates up or down the max rate value
        # these will be clipped to the bounds of the simulator
        self.action_space = gym.spaces.Box(
            low=-np.array(list(self.net.get_controlled_rates(self.player_name).values()), dtype=np.float32),
            high=self.max_rate - np.array(list(self.net.get_controlled_rates(self.player_name).values()),
                                          dtype=np.float32),
            dtype=np.float32)

    def generate_action(self, action, current_rates):
        """
        Generate the new rates from the desired rate adjustments
        Results are rounded to better allow caching
        :param action: Rate adjustments
        :param current_rates: current rates from the net
        :return: new rate values
        """
        return np.around(np.clip(current_rates + action, 0, self.max_rate), 2)

    def step(self, action):
        """
        Step the environment updating the rate corresponding to the desired rate adjustments
        :param action: Rate adjustments
        :return: response from the environment
        """
        rates = self.generate_action(action, self.last_rates)
        # we assume the rate adjust always worked
        self.last_rates = np.array(rates)
        return self.env.step(rates)

    def reset(self):
        """
        Reset the environment
        :return: response from the environment with the wrapper applied
        """
        obs = self.env.reset()
        # TODO optimize from obs
        self.last_rates = np.array(list(self.env.get_controlled_rates().values()))
        return obs
