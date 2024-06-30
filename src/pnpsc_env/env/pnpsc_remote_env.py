import numpy as np
import requests

from .pnpsc_env import PnpscEnv


class PnpscRemoteEnv(PnpscEnv):
    """
    Remote implementation of the PnpscEnv abstract class
    Uses the cloud simulator provided by Colvette
    """

    def __init__(self, player_name, net_path, sim_url='http://pnpsc.net:8001/', max_tokens=16, max_rate=10):
        """
        Create a wrapper for the PNPNSC simulator
        :param player_name: Name of the agent player, must match one of the players in the PNPSC net definition
        :param net_path: Path to the PNPSC net definition
        :param sim_url: URL of the simulator
        :param max_tokens: Maximum expected tokens at any place
        :param max_rate: Maximum rate allowed a at any transition
        """

        self.sim_url = sim_url
        super().__init__(player_name, net_path, max_tokens, max_rate)

    def _update_simulator(self, action, player_name):
        """
        Perform a transition rate update and step the simulator one step
        :param action: Player's rate updates
        :return: Next observation and reward from the environment
        """
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

        current_rates = list(self.net.get_controlled_rates(player_name).values())
        update_cost = float(self.c_change(np.array(action), np.array(current_rates)))

        updates = [{'name': t[0], 'rate': t[1]} for t in self.net.get_controlled_rates(player_name).items()]
        i = 0
        for k, v in self.net.get_controlled_rates(player_name).items():
            if v != action[i]:
                updates[i]['rate'] = float(action[i])
            i += 1

        # Build the json update object
        costs = [{'name': player_name, 'transition_change_cost': update_cost}]
        data = {'players': costs, 'transitions': updates}

        # Send the update request
        res = requests.post(self.sim_url + 'change_transitions/', json=data, headers=headers)
        if res.status_code != 200 or not res.json()['changes_made']:
            print('error updating rates')

    def _step_simulator(self):
        """
        Step the simulator
        """
        res = requests.get(self.sim_url + 'step/')
        if res.status_code != 200:
            print('error stepping net', res.json())

        # Parse the results
        self.net.update_net(res.json())

    def _reset_simulator(self):
        """
        Reset the simulator
        """
        # Delete the old PNPSC net definition
        res = requests.get(self.sim_url + 'delete/')
        if res.status_code != 200:
            print('failed deleting net', res)

        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

        # Upload the provided PNPSC net definition
        res = requests.post(self.sim_url + 'uploadpetrinet/', json=self.net.get_json(), headers=headers)
        if res.status_code != 200:
            print('failed uploading net', res)

        # Get the initial state from the simulator
        res = requests.get(self.sim_url + 'status/')
        self.net.update_net(res.json())

    def render(self):
        pass
