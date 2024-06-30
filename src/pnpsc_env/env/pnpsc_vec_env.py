import gym
import numpy as np

from .pnpsc_env import PnpscEnv
from .pnpsc_local_env import PnpscLocalEnv


# TODO specialize for 1 env
class PnpscVecEnv(PnpscEnv):

    def __init__(self, player_name, net_path, max_tokens=10, max_rate=10,
                 num_envs=1):
        """
        Create a wrapper for the PNPNSC simulator
        :param player_name: Name of the agent player, must match one of the players in the PNPSC net definition
        :param net_path: Path to the PNPSC net definition
        :param max_tokens: Maximum expected tokens at any place
        :param max_rate: Maximum expected rate for any transition
        :param num_envs: Number of parallel executions in each step
        """
        super().__init__(player_name, net_path, max_tokens, max_rate)
        self.penalty = None
        self.num_envs = num_envs
        self.net_path = net_path

        self.other_players = []
        self.other_strategies = {}

        self.obs_places = {}
        self.goal_places = []
        self.end_places = []
        for player in self.net.players:
            obs_places = []
            for i, p in enumerate(self.net.get_all_places()):
                if p in self.net.get_visible_places(player):
                    obs_places.append(i)
            self.obs_places[player] = np.array(obs_places)

        for i, p in enumerate(self.net.get_all_places()):
            if p in self.net.get_goal_places(player_name):
                self.goal_places.append(i)
            if p in self.net.get_end_places(player_name):
                self.end_places.append(i)

        self.goal_places = np.array(self.goal_places)
        self.end_places = np.array(self.end_places)

        self.obs_rates = {}
        for player in self.net.players:
            obs_rates = []
            for i, p in enumerate(self.net.get_all_rates()):
                if p in self.net.get_controlled_rates(player):
                    obs_rates.append(i)
            self.obs_rates[player] = np.array(obs_rates)

        # Observation space is all places visible to the player and transition rates controlled by player
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.zeros(len(self.obs_places[player_name]) + len(self.obs_rates[player_name]))],
                               dtype=np.float32),
            high=np.concatenate([np.full(len(self.obs_places[player_name]), max_tokens, dtype=np.float32),
                                 np.full(len(self.obs_rates[player_name]), max_rate)]), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=self.max_rate, shape=(len(self.obs_rates[player_name]),),
                                           dtype=np.float32)

        self.t = 0

        self.places = np.array([p for p in list(self.net.get_all_places().values())])
        self.rates = np.array([r for r in list(self.net.get_all_rates().values())])

        # build matrix for graph operations
        enabled_mask = []
        inhibitor_mask = []
        input_mask = []
        output_mask = []
        control_rates = []

        places_names = self.net.get_all_places().keys()
        for t in sorted(self.net.json['transitions'], key=lambda x: x['name']):
            input_places = t['input'].split(',')
            output_places = t['output'].split(',')
            inhibitor_places = t['inhibitor'].split(',')
            cr = {r2[0]: float(r2[1]) for r2 in [r.split('=') for r in t['control_rate'].split(',')] if len(r2) > 1}
            enabled_mask.append([1 if p in input_places else 0 for p in places_names])
            inhibitor_mask.append([1 if p in inhibitor_places else 0 for p in places_names])
            input_mask.append([1 if p in input_places else 0 for p in places_names])
            output_mask.append([1 if p in output_places else 0 for p in places_names])
            control_rates.append([cr[p] if p in cr else 0 for p in places_names])

        self.enabled_mask = np.array(enabled_mask).T
        self.inhibitor_mask = np.array(inhibitor_mask).T
        self.num_in_transitions = np.sum(self.enabled_mask, 0)
        self.input_mask = np.array(input_mask)
        self.output_mask = np.array(output_mask)
        self.control_rates = np.array(control_rates).T

        self.last_mean_reward = None

    def _reset_simulator(self):
        pass

    def render(self):
        pass

    def _step_simulator(self):
        pass

    def _update_simulator(self, action, player_name):
        pass

    #@cache
    def _run_batch_until_complete(self, places, rates):
        """
        Run the current net definition to completion (with no further action by any players)
        :param places: current marking
        :param rates: current rates
        :return: the mean reward
        """
        places = np.repeat(np.array(places)[:, np.newaxis], self.num_envs, axis=1).T
        rates = np.array(rates)

        # apply opponent strategy
        for i, k in enumerate(self.net.get_all_rates()):
            if k in self.other_strategies:
                rates[i] = self.other_strategies[k]
        rewards = np.zeros(self.num_envs).reshape(-1, 1)

        dones = [False]
        while not all(dones):
            enabled = np.matmul(np.clip(places, 0, 1), self.enabled_mask) // self.num_in_transitions
            enabled &= np.invert(np.matmul(np.clip(places, 0, 1), self.inhibitor_mask))

            temp_rates = (np.matmul(np.clip(places, 0, 1), self.control_rates) + rates) * enabled * np.invert(dones)

            # If only player transitions are enabled and they all have rate 0, we end the episode
            dones = np.invert(np.any(temp_rates, axis=1).reshape(-1, 1))

            with np.errstate(divide='ignore'):
                ft = np.random.exponential(1 / temp_rates)

            j = np.argmin(ft, axis=1)
            # selected transition to fire

            all_disabled = np.invert(np.all(np.isinf(ft), axis=1).reshape(-1, 1))
            places -= self.input_mask[j] * np.invert(dones) * all_disabled
            places += self.output_mask[j] * np.invert(dones) * all_disabled

            if len(self.goal_places > 0):
                rewards += 100 * np.clip(np.sum(np.take(places, self.goal_places, axis=1), axis=1), 0, 1).reshape(-1,
                                                                                                                  1) * np.invert(
                    dones)
                dones |= np.any(np.take(places, self.goal_places, axis=1), axis=1).reshape(-1, 1)

            # Check if any end places are marked and end simulation
            if len(self.end_places > 0):
                dones |= np.any(np.take(places, self.end_places, axis=1), axis=1).reshape(-1, 1)

        return np.mean(rewards)

    def step(self, action, step_sim=True):
        """
        Step the environment with the player's action
        :param action: Player's rate updates
        :param step_sim: Allow the player to update rates without stepping the simulator
        :return: environment observation and reward
        """
        if self.last_mean_reward is None:
            self.last_mean_reward = self._run_batch_until_complete(tuple(self.places), tuple(self.rates))
        # Update the rates if provided
        if action is not None:
            action = np.array(action)
            action.clip(0, self.max_rate, action)
            player_rates = np.take(self.rates, self.obs_rates[self.player_name])
            reward = - self.c_change(action, player_rates)
            np.put(self.rates, self.obs_rates[self.player_name], action)
        else:
            reward = 0

        # Only step the sim if requested, used to allow multi-action players
        if step_sim:
            enabled = np.matmul(np.clip(self.places, 0, 1), self.enabled_mask) // self.num_in_transitions
            enabled &= np.invert(np.matmul(np.clip(self.places, 0, 1), self.inhibitor_mask))
            # dones = np.invert(np.any(enabled))

            temp_rates = (np.matmul(np.clip(self.places, 0, 1), self.control_rates) + self.rates) * enabled

            # If only player transitions are enabled and they all have rate 0, we end the episode
            done = np.invert(np.any(temp_rates))

            with np.errstate(divide='ignore'):
                ft = np.random.exponential(1 / temp_rates)

            j = np.argmin(ft)
            # selected transition to fire
            self.t += np.min(ft) if not done else 0

            all_disabled = np.invert(np.all(np.isinf(ft)))
            self.places -= self.input_mask[j] * np.invert(done) * all_disabled
            self.places += self.output_mask[j] * np.invert(done) * all_disabled

            if len(self.goal_places > 0):
                reward += 100 * np.clip(np.sum(np.take(self.places, self.goal_places)), 0, 1)
                done |= np.any(np.take(self.places, self.goal_places))

            # Check if any end places are marked and end simulation
            if len(self.end_places > 0):
                done |= np.any(np.take(self.places, self.end_places))

            if done:
                reward -= self.last_mean_reward
            else:
                # let the other player's act
                for p in self.other_players:
                    a = np.clip(p.act(self.net)[0], 0, self.max_rate)
                    if a is not None:
                        # player_rates = np.take(self.rates, self.obs_rates[p])
                        # reward = -self.c_change(action - player_rates)
                        np.put(self.rates, self.obs_rates[p.player_name], a)
        else:
            done = False

        if not done:
            current_mean_reward = self._run_batch_until_complete(tuple(self.places), tuple(self.rates))

            reward += current_mean_reward - self.last_mean_reward
            self.last_mean_reward = current_mean_reward

        for i, k in enumerate(self.net.places.keys()):
            self.net.places[k] = self.places[i]
        for i, k in enumerate(self.net.rates.keys()):
            self.net.rates[k] = self.rates[i]

        return self.get_observation(self.player_name), reward, done, {}

    def reset(self, info=False):
        """
        Reset the environment
        :param info: Should the debugging info be returned?
        :return: the initial observation
        """
        self.last_mean_reward = None
        self.t = 0

        self.net.costs = {p: 0 for p in self.net.players}
        self.net.places = {p['name']: p['marking'] for p in sorted(self.net.json['places'], key=lambda x: x['name'])}
        self.net.rates = {t['name']: t['rate'] for t in sorted(self.net.json['transitions'], key=lambda x: x['name'])}

        self.places = np.array([p for p in list(self.net.get_all_places().values())])
        self.rates = np.array([r for r in list(self.net.get_all_rates().values())])

        return self.get_observation(self.player_name)

    def get_observation(self, player_name):
        """
        Generates the observation for a player
        :param player_name: name of the player for whom the observation is for
        :return: the observation and reward for the player
        """
        return np.concatenate([np.take(self.places, self.obs_places[player_name]),
                               np.take(self.rates, self.obs_rates[player_name])])

    def get_controlled_rates(self):
        """
        Gets the rates controlled by the current player
        :return: A dictionary of the rates
        """
        # TODO optimize this
        # update if we are not maintaining it
        return self.net.get_controlled_rates(self.player_name)

    def add_other_player(self, agent):
        """
        Add another player model to the PNPSC net simulator
        :param agent: An AbstractAgent
        """
        self.other_players.append(agent)
        # TODO eval strategy and add to rates
        # merge the strategies, no rate can be controlled by more than one player
        #self.other_strategies.update(self._eval_strategy(agent, num_runs=10_000))

    def update_strategies(self):
        for p in self.other_players:
            self.other_strategies.update(self._eval_strategy(p, num_runs=10_000))

    def _eval_strategy(self, agent, num_runs=10_000):
        """
        Evaluate the agents strategy to determine the mean updated final rates.
        This is used to approximate the opponents strategy to greatly speed up execution
        :param agent: agent to evaluate strategy
        :param num_runs: number of executions
        :return: A dictionary of ending rates the agent updated
        """
        end_places = []
        for i, k in enumerate(self.net.places):
            if i in self.goal_places or i in self.end_places:
                end_places.append(k)
        print("evaluating strategy for:", agent.player_name)
        env = PnpscLocalEnv(agent.player_name, self.net_path, max_tokens=self.max_tokens)

        start_rates = dict.copy(env.net.rates)
        end_rates = {}
        for k in env.net.rates:
            end_rates[k] = []

        for i in range(num_runs):
            env.reset()
            done = False
            while not done:
                _, _, done, _ = env.step(agent.act(env.net)[0])

            for k, v in env.net.rates.items():
                if start_rates[k] != v:
                    end_rates[k].append(v)

        strategy = {}
        for k, v in end_rates.items():
            if len(v) > 0:
                strategy[k] = np.mean(end_rates[k])

        return strategy
