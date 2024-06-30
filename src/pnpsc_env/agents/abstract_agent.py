from abc import ABC, abstractmethod

import numpy as np


class AbstractAgent(ABC):
    """
    An agent that simulates a player for a PNPSC net.
    """
    def __init__(self, player_name):
        """
        :param player_name: Name of the player, must match a player listed in the net definition
        """
        self.player_name = player_name

    def act(self, net, output_strategy=False):
        """
        Performs the player action for the given state of the net
        :param net: the current pnpsc net object
        :param print_strategy: toggle to output the strategy of the player
        :return: The desired rates for the player controlled transitions
        """
        rates = self._act(net)

        strategy = None
        if output_strategy:
            rate_updates = np.array(rates) - np.array(list(net.get_controlled_rates(self.player_name).values()))
            np.clip(rate_updates, 0, self.env.max_rate, rate_updates)
            if sum(rate_updates) != 0:
                rate_names = list(net.get_controlled_rates(self.player_name).keys())
                strategy = {tuple([k for k, v in net.get_visible_places(self.player_name).items() if v != 0]):
                                {rate_names[i]: v for i, v in enumerate(rate_updates) if v != 0}}

        return rates, strategy

    """
    Implementation of the agent's action (do not call directly)
    :param net: the current pnpsc net object
    :return: The desired rates for the player controlled transitions
    """
    @abstractmethod
    def _act(self, net):
        pass

