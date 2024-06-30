import numpy as np

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """
    A PNPSC player agent that updates a single random rate each step to a random value between 0 and 10
    """
    def __init__(self, player_name, eps=1.0):
        """
        :param player_name: Name of the player, must match a player listed in the net definition
        :param eps: optional probability to choose a random action (otherwise no action)
        """
        super().__init__(player_name)
        self.eps = eps

    def _act(self, net, print_strategy=False):
        """
        Performs the player action for the given state of the net
        :param net: the current pnpsc net object
        :param print_strategy: toggle to output the strategy of the player
        :return: The desired rates for the player controlled transitions
        """
        rates = list(net.get_controlled_rates(self.player_name).values())
        if np.random.sample() < self.eps:
            i = np.random.choice(len(rates))
            rates[i] = np.random.choice(10)
        return rates
