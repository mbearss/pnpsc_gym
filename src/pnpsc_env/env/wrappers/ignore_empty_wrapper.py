import gym
import numpy as np


class IgnoreEmptyWrapper(gym.Wrapper):
    """
    Wrapper to the PNPSC environment to skip actions with no markings present
    """
    def __init__(self, env):
        """
        Create a wrapper for the PNPSC environment
        :param env: PNPSC environment to wrap
        """
        super().__init__(env)
        self.env = env
        self.visible_place_count = len(env.net.get_visible_places(self.player_name))

    def step(self, action, step_sim=True):
        """
        Step the environment one time with the desired action.
        If the observation contains no visible places that are marked, continue to step the environment until
        done is True or an observation is observed with a visible marking
        :param action: Action to perform
        :return: response from the environment
        """
        reward = 0
        next_state, r, done, info = self.env.step(action, step_sim)
        reward += r
        while np.all(next_state[:self.visible_place_count] == 0) and not done:
            # no visible marking
            next_state, r, done, info = self.env.step(None)
            reward += r
        return next_state, reward, done, info

    def reset(self):
        """
        Reset the environment.
        If the observation contains no visible places that are marked, step the environment until
        done is True or an observation is observed with a visible marking
        :return: response from the environment
        """
        next_state, done = self.env.reset(), False
        while np.all(next_state[:self.visible_place_count] == 0) and not done:
            # no visible marking
            next_state, reward, done, info = self.env.step(None)
        return next_state
