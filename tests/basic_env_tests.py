import json
import unittest

import numpy as np

from src.pnpsc_env.env.pnpsc_vec_env import PnpscVecEnv
from src.pnpsc_env.agents.static_agent import StaticAgent
from src.pnpsc_env.env.pnpsc_local_env import PnpscLocalEnv
from src.pnpsc_env.env.pnpsc_net import PnpscNet
from src.pnpsc_env.env.pnpsc_remote_env import PnpscRemoteEnv


class TestEnvMethods(unittest.TestCase):

    def test_local_env(self):
        """
        Test the basic functionality of the local environment
        """
        env = PnpscLocalEnv(player_name='Attacker', net_path='../../nets/example_net.json')
        agent = StaticAgent(player_name='Attacker')

        state, done = env.reset(), False
        i = 0
        while not done and i < 100:
            state, reward, done, info = env.step(agent.act(env.net)[0])
            i += 1

        self.assertTrue(done)

    def test_remote_env(self):
        """
        Test the basic functionality of the remote environment
        """
        env = PnpscRemoteEnv(player_name='Attacker', net_path='../../nets/example_net.json')
        agent = StaticAgent(player_name='Attacker')

        state, done = env.reset(), False
        i = 0
        while not done and i < 100:
            state, reward, done, info = env.step(agent.act(env.net)[0])
            i += 1

        self.assertTrue(done)

    def test_vec_env(self):
        """
        Test the basic functionality of the vectorized environment
        """
        env = PnpscVecEnv(player_name='Attacker', net_path='../../nets/example_net.json', num_envs=1)
        agent = StaticAgent(player_name='Attacker')

        state, done = env.reset(), False
        i = 0
        while not done and i < 100:
            state, reward, done, info = env.step(agent.act(env.net)[0])
            i += 1

        self.assertTrue(done)


    def test_read_net(self):
        """
        Test reading a PNPSC net definition from a file
        """
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)

            self.assertEqual(net.players, ['Attacker', 'Defender'])
            self.assertEqual(net.get_all_places(), {'aP1': 10, 'aP2': 0, 'aP3': 0, 'aP4': 0, 'aP5': 0})
            self.assertEqual(net.get_all_rates(), {'aT1': 10, 'aT2': 5, 'aT3': 10, 'aT4': 2})

            self.assertEqual(net.get_visible_places('Attacker'), {'aP1': 10})
            self.assertEqual(net.get_visible_places('Defender'), {})
            self.assertEqual(net.get_controlled_rates('Attacker'), {'aT1': 10})
            self.assertEqual(net.get_controlled_rates('Defender'), {})

            self.assertEqual(net.get_player_cost('Attacker'), 0)
            self.assertEqual(net.get_player_cost('Defender'), 0)

    def test_update_net(self):
        """
        Test parsing a JSON response and updating the PNPSC net definition
        """
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)

            self.assertEqual(net.get_controlled_rates('Attacker'), {'aT1': 10})
            with open('test_response.json') as res:
                net.update_net(json.load(res))
            self.assertEqual(net.get_all_places(), {'aP1': 9, 'aP2': 1, 'aP3': 1, 'aP4': 0, 'aP5': 0})
            self.assertEqual(net.get_player_cost('Attacker'), 10)

    def test_action(self):
        """
        Test a single action from an agent
        """
        env = PnpscLocalEnv(player_name='Attacker', net_path='../../nets/example_net.json')
        env.c_change = lambda x, y: np.sum(np.abs(x - y))
        self.assertEqual(env.net.get_controlled_rates('Attacker'), {'aT1': 10})
        env.step([5])
        self.assertEqual(env.net.get_controlled_rates('Attacker'), {'aT1': 5})
        self.assertEqual(env.net.get_player_cost('Attacker'), 5)


if __name__ == '__main__':
    unittest.main()
