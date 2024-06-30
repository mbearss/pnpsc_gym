import gym
from collections.abc import Mapping
import numpy as np

from ...simulator.simulator import Simulator


class FrozenDict(Mapping):
    """Frozen dictionary implementation to record various states seen during execution"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __str__(self):
        s = '{'
        for k, v in self._d.items():
            s += ('\'' if type(s) is str else '') + str(k) + ('\'' if type(s) is str else '') + ': ' + str(v) + ', '
        return s[:-2] + '}'

    def __hash__(self):
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash


class MarkingRecorder(gym.Wrapper):
    """
    Wrapper to the PNPSC environment to allow recording of marking seen during execution
    """
    def __init__(self, env):
        """
        Create a wrapper for the PNPSC environment
        :param env: PNPSC environment to wrap
        """
        super().__init__(env)
        self.env = env
        self.markings = {FrozenDict(self.env.net.get_visible_places(self.env.player_name)): 1}

    def step(self, action):
        """
        Step the simulator and record the marking
        :param action: Action to perform
        :return: response from the environment
        """
        next_state, reward, done, info = self.env.step(action)
        state = FrozenDict(self.env.net.get_visible_places(self.env.player_name))
        if state in self.markings:
            self.markings[state] += 1
        else:
            self.markings[state] = 1
        return next_state, reward, done, info

    def reset(self):
        """
        Reset the environment
        :return: Response from the environment
        """
        next_state = self.env.reset()
        state = FrozenDict(self.env.net.get_visible_places(self.env.player_name))
        if state in self.markings:
            self.markings[state] += 1
        else:
            self.markings[state] = 1
        return next_state

    def get_seen_marked_places(self):
        """
        Get a set of all observed markings
        :return: set of observed markings
        """
        one_markings = set()
        for m in self.markings.keys():
            s = [k for k, v in m.items() if v > 0]
            one_markings.add(tuple(s))
        return sorted(one_markings)

    def get_seen_enabled_transitions(self):
        """
        Get a set of all enabled transitions
        :return: set of observed enabled trasntisions
        """
        one_enabled = set()
        sim = Simulator(self.env.net)
        for marking in self.get_seen_marked_places():
            sim.reset()
            self.env.net.places = {k: 0 for k in self.env.net.places.keys()}
            for m in marking:
                self.env.net.places[m] = 1
            enabled = np.array(list(self.env.net.get_all_rates().keys()))[sim._check_enabled()]
            one_enabled.add(tuple(enabled))
        return sorted(one_enabled)

