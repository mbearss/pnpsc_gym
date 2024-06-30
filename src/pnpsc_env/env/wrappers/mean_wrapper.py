from functools import lru_cache
import gym
import numpy as np

from .marking_recorder import FrozenDict


class MeanWrapper(gym.Wrapper):
    """
    """
    def __init__(self, env, num_runs=100):
        """
        DEPRECATED
        Create a wrapper for the PNPSC environment
        :param env: PNPSC environment to wrap
        """
        super().__init__(env)
        self.env = env
        self.num_runs = num_runs
        self.places = None
        self.rates = None
        self.last_mean_reward = None

    @lru_cache(maxsize=None)
    def calc_mean_reward(self, places, rates):
        rewards = []
        costs = FrozenDict(self.env.net.costs)
        for i in range(self.num_runs):
            rewards.append(self.env.run_until_complete())
            # reset net state
            for k in self.env.net.places.keys():
                self.env.net.places[k] = places[k]
            #for k in self.env.net.rates.keys():
            #    self.env.net.rates[k] = rates[k]
            for k in self.env.net.costs.keys():
                self.env.net.costs[k] = costs[k]
            self.env.net.done = False
        return np.mean(rewards)

    def step(self, action, step_sim=True):
        """
        :return: next state, reward, done, debug info
        """
        if self.last_mean_reward is None:
            self.last_mean_reward = self.calc_mean_reward(self.places, self.rates)

        next_state, reward, done, info = self.env.step(action, step_sim)
        self.places = FrozenDict(self.env.net.get_all_places())
        self.rates = FrozenDict(self.env.net.get_all_rates())

        if done:
            reward -= self.last_mean_reward
        else:
            current_mean_reward = self.calc_mean_reward(self.places, self.rates)
            # agent reward is the difference in mean reward
            reward += current_mean_reward - self.last_mean_reward
            self.last_mean_reward = current_mean_reward

        return next_state, reward, done, info

    def reset(self):
        """
        :return: The initial state of the simulator
        """
        next_state = self.env.reset()
        self.places = FrozenDict(self.env.net.get_all_places())
        self.rates = FrozenDict(self.env.net.get_all_rates())
        self.last_mean_reward = None
        return next_state

    def render(self):
        """
        Forward the render command to the environment
        :return:
        """
        self.env.render()
