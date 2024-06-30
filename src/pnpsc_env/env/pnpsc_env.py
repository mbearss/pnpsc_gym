import json
import os
from abc import ABC, abstractmethod

import gym
import numpy as np

from .pnpsc_net import PnpscNet
from ..agents.abstract_agent import AbstractAgent


class PnpscEnv(ABC, gym.Env):

    def __init__(self, player_name, net_path, max_tokens=10, max_rate=10):
        """
        Create a Gym environment that wraps for the PNPNSC simulator
        :param player_name: Name of the agent player, must match one of the players in the PNPSC net definition
        :param net_path: Path to the PNPSC net definition
        :param max_tokens: Maximum expected tokens at any place
        :param max_rate: Maximum expected rate for any transition
        """
        self.player_name = player_name
        self.max_tokens = max_tokens
        self.max_rate = max_rate

        self.other_players = []

        self.net_path = net_path

        # Load the PNPSC definition from path provided
        with open(os.getcwd() + '/nets/' + net_path) as f:
            data = json.load(f)
            self.net = PnpscNet(data)

        self.goal_places = self.net.get_goal_places(player_name)
        self.end_places = self.net.get_end_places(player_name)

        # Verify the player name is valid
        assert self.player_name in self.net.players, 'player_name must be part of PNPSC net definition'

        # Observation space is all places visible to the player and transition rates controlled by player
        places = len(self.net.get_visible_places(self.player_name))
        rates = len(self.net.get_controlled_rates(self.player_name))

        # for exponential cost
        self.original_rates = np.array(list(self.net.get_controlled_rates(self.player_name).values()))

        self.observation_space = gym.spaces.Box(low=0, high=self.max_tokens, shape=((places + rates),), dtype=float)
        self.action_space = gym.spaces.Box(low=0, high=self.max_rate, shape=(rates,), dtype=float)

        # Accumulated costs so far, used to find the current action cost
        self.last_cost = 0

    def c_change(self, a, cr):
        """
        An arbitrary cost function for testing
        :param action: Rate update distance
        :param cr: Current rates
        :return: Cost incurred by update
        """
        return np.sum(np.abs(a - cr)) / 10

    def _pre_step(self, action):
        """
        Action to perform before the players rates are set
        :param action: player's rate updates
        """
        # clip the rates to the simulators bounds
        action.clip(0, self.max_rate, action)

    def _post_step(self, action):
        """
        Action to perform after the players rates are set
        :param action: player's rate updates
        """
        # perform other player's actions
        for p in self.other_players:
            state, _, _, _ = self.get_observation(p.player_name)
            a = np.clip(p.act(self.net)[0], 0, self.max_rate)
            self._update_simulator(a, p.player_name)

    @abstractmethod
    def _update_simulator(self, action, player_name):
        """
        Perform the rate updates to the simulator
        :param action: player's rate updates
        :param player_name: player to update
        """
        pass

    @abstractmethod
    def _reset_simulator(self):
        """
        Reset the simulator to the initial configuration specified in the definition file
        """
        pass

    def step(self, action, step_sim=True):
        """
        Step the environment with the player's action
        :param action: Player's rate updates
        :param step_sim: Allow the player to update rates without stepping the simulator
        :return: environment observation and reward
        """
        if action is not None:
            action = np.array(action)
            self._pre_step(action)
            self._update_simulator(action, self.player_name)

        if step_sim:
            self._post_step(action)
            self._step_simulator()
        return self.get_observation(self.player_name)

    def get_observation(self, player_name):
        """
        Generates the observation for a player
        :param player_name: name of the player for whom the observation is for
        :return: the observation and reward for the player
        """
        state = np.concatenate([list(self.net.get_visible_places(player_name).values()),
                                list(self.net.get_controlled_rates(player_name).values())])

        reward = 0
        # Only tracking the current learning player right now
        if self.player_name == player_name:
            reward = -(self.net.get_player_cost(player_name) - self.last_cost)
            # Update the cost tracking variable
            self.last_cost = self.net.get_player_cost(player_name)

        all_places = self.net.get_all_places()
        done = self.net.done
        for goal in self.goal_places:
            if all_places[goal] > 0:
                done = True
                reward += 100
        for end in self.end_places:
            if all_places[end] > 0:
                done = True

        return state, reward, done, {'places': self.net.get_all_places()}

    def reset(self, info=False):
        """
        Reset the environment
        :param info: Should the debugging info be returned?
        :return: the initial observation
        """
        self._reset_simulator()
        self.last_cost = 0

        state = np.concatenate([list(self.net.get_visible_places(self.player_name).values()),
                                list(self.net.get_controlled_rates(self.player_name).values())])

        if info:
            return state, {'places': self.net.get_all_places()}
        else:
            return state

    @abstractmethod
    def render(self):
        """
        Optional method to provide a visual representation of the environment
        """
        pass

    def add_other_player(self, agent: AbstractAgent):
        """
        Add another player model to the PNPSC net simulator
        :param agent: An AbstractAgent
        """
        self.other_players.append(agent)

    @abstractmethod
    def _step_simulator(self):
        """
        Step the simulator
        """
        pass

    def run_until_complete(self):
        """
        Run the net to completion and return the total reward
        :return: the total reward received from this current point
        """
        reward, done = 0, False
        while not done:
            _, r, done, _ = self.step(None)
            reward += r
        return reward

    def get_controlled_rates(self):
        """
        Gets the rates controlled by the current player
        :return: A dictionary of the rates
        """
        return self.net.get_controlled_rates(self.player_name)
