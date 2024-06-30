import unittest

from src.pnpsc_env.agents.static_agent import StaticAgent
from src.pnpsc_env.env.pnpsc_local_env import PnpscLocalEnv
from src.pnpsc_env.env.wrappers.marking_recorder import MarkingRecorder


class TestEnvMethods(unittest.TestCase):

    def test_marking_recorder(self):
        """
        Test the basic functionality of the local environment
        """
        env = PnpscLocalEnv(player_name='Attacker', net_path='../../nets/example_net.json')
        env = MarkingRecorder(env)
        agent = StaticAgent('Attacker')

        state, done = env.reset(), False
        i = 0
        while not done and i < 100:
            state, reward, done, info = env.step(agent.act(env.net)[0])
            i += 1

        assert len(env.markings) > 0


if __name__ == '__main__':
    unittest.main()
