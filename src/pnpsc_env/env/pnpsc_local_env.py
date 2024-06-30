import numpy as np

from .pnpsc_env import PnpscEnv
from ..simulator.simulator import Simulator


class PnpscLocalEnv(PnpscEnv):
    """
    Local implementation of the PnpscEnv abstract class
    """

    def __init__(self, player_name, net_path, max_tokens=16, max_rate=10):
        """
        Create a wrapper for the PNPNSC simulator
        :param player_name: Name of the agent player, must match one of the players in the PNPSC net definition
        :param net_path: Path to the PNPSC net definition
        :param max_tokens: Maximum expected tokens at any place
        :param max_rate: Maximum rate allowed a at any transition
        """
        super().__init__(player_name, net_path, max_tokens, max_rate)

        # Load the PNPSC definition from path provided
        self.simulator = Simulator(self.net)

    def _update_simulator(self, action, player_name):
        """
        Perform a transition rate update and step the simulator one step
        :param action: Player's rate updates
        :return: Next observation and reward from the environment
        """
        current_rates = list(self.net.get_controlled_rates(player_name).values())
        update_cost = self.c_change(np.array(action), np.array(current_rates))

        updates = {}
        i = 0
        for k, v in self.net.get_controlled_rates(player_name).items():
            if v != action[i]:
                updates[k] = action[i]
            i += 1

        self.simulator.update_rates(updates)
        self.net.costs[player_name] += update_cost

    def _step_simulator(self):
        """
        Step the simulator
        """
        self.simulator.step()

    def _reset_simulator(self):
        """
        Reset the simulator
        """
        self.simulator.reset()

    def render(self):
        """
        Render the current PNPSC net
        """
        self.simulator.render()

    def output_net(self, path, show_desc=False):
        self.simulator.output_net(path, show_desc)
