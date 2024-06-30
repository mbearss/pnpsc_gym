import gym
import numpy as np


class DiscretePnpscWrapper(gym.Wrapper):
    """
    Wrapper to the PNPSC environment to allow a choice of a single discrete update each step
    """

    def __init__(self, env, f, options, max_actions):
        """
        Create a wrapper for the PNPSC environment
        :param env: PNPSC environment to wrap
        :param f: update function for rates
        :param options: update options for f
        :param max_actions: maximum updates each simulator step
        """
        super().__init__(env)
        self.env = env
        self.f = f
        self.max_actions = max_actions
        self.action = 0

        # used to track duplicate actions
        self.action_this_turn = []

        # The action table enumerates all possible actions, the agent can manipulate any transition under its control
        # by multiplying it by one of a selection of values. Only one rate can be updated each step in the env
        self.actions_table = []
        for i in self.env.net.get_controlled_rates(self.env.player_name):
            for j in options:
                self.actions_table.append((i, j))
        self.actions_table.append(('end', 0))

        obs_places = self.env.net.get_visible_places(self.player_name)
        obs_rates = self.env.net.get_controlled_rates(self.player_name)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(len(obs_places) + len(obs_rates) + 1)], dtype=np.float32),
            high=np.concatenate([np.full(len(obs_places), self.env.max_tokens, dtype=np.float32),
                                 np.full(len(obs_rates), self.env.max_rate, dtype=np.float32), [self.max_actions]]),
            dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.actions_table))

    def step(self, action):
        """
        Step the environment updating the rate corresponding to the discrete action selected by the agent
        :param action: Action in the action table
        :return: response from the environment with an optionally added penalty
        """
        self.action = (self.action + 1) % self.max_actions
        if self.action == 0:
            self.action_this_turn = []
        # end turn action
        if action == len(self.actions_table) - 1:
            next_state, reward, done, info = self.env.step(None)
            next_state = np.append(next_state, self.action)
            self.action = 0
        else:
            # Only step the sim if we are at the max action count
            rates = self.generate_action(action, self.env.get_controlled_rates())
            next_state, reward, done, info = self.env.step(list(rates.values()), step_sim=(self.action == 0))
            next_state = np.append(next_state, self.action)
            self.action_this_turn.append(action)
        return next_state, reward, done, info

    def generate_action(self, action, rates):
        """
        Applies the desired action from the action table to the supplied rates
        :param action: desired action
        :param rates: rates to update
        :return: the updated rates
        """
        # last action is skip turn
        if action == len(self.actions_table) - 1:
            return None
        # Update transition ti by applying f(current_rate, option_value)
        ti, to = self.actions_table[action]
        rates[ti] = self.f(rates[ti], to)
        return rates

    def reset(self):
        """
        Reset the environment
        :return: response from the environment with the wrapper applied
        """
        self.action = 0
        self.action_this_turn = []
        next_state = self.env.reset()
        next_state = np.append(next_state, self.action)
        return next_state
