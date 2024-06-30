import numpy as np

from .abstract_agent import AbstractAgent


class Capec63Agent(AbstractAgent):
    """
    A reimplementation of the optimal strategy for CAPEC-63 from [Bland, 2020]. This agent is
    only assumed to work for the exact CAPEC-63 net structure and rates found in the cited paper.
    """
    def __init__(self, player_name, eps=0.0):
        """
        :param player_name: Name of the player, must match a player listed in the net definition
        :param eps: optional probability to choose a random action instead of the optimal one
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
        places = net.get_all_places()
        rates = net.get_controlled_rates(self.player_name)

        # select a random update with probability eps
        if np.random.sample() < self.eps:
            i = np.random.choice(len(rates))
            rates = list(rates.values())
            rates[i] = np.random.choice([0, 10])
            return rates

        if self.player_name == 'Attacker':
            if places['aP2'] >= 1:
                _set_all_rates(rates, ['aT2', 'aT5', 'aT8'], 10)
            elif places['aP10'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT2', 'bT5', 'bT8', 'bT11'], 10)
            elif places['aP7'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT2', 'bT5', 'bT8', 'bT11'], 10)
            elif places['aP10'] >= 1 and places['bP7'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT2', 'cT5', 'cT8', 'cT14'], 10)
                _set_all_rates(rates, ['cT11'], 0)
            elif places['aP4'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT2', 'bT5', 'bT8', 'bT11'], 10)
            elif places['aP10'] >= 1 and places['bP10'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT2', 'cT5', 'cT8', 'cT14'], 10)
                _set_all_rates(rates, ['cT11'], 0)
            elif places['aP10'] >= 1 and places['bP3'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT2', 'cT5'], 10)
                _set_all_rates(rates, ['cT8', 'cT11', 'cT14'], 0)
            elif places['aP10'] >= 1 and places['bP13'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT2', 'cT5', 'cT8', 'cT14'], 10)
                _set_all_rates(rates, ['cT11'], 0)
            elif places['aP4'] >= 1 and places['bP7'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT2', 'cT5', 'cT8'], 10)
                _set_all_rates(rates, ['cT11', 'cT14'], 0)
            elif places['aP10'] >= 1 and places['bP7'] >= 1 and places['cP7'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT6', 'dT15'], 10)
                _set_all_rates(rates, ['dT2', 'dT16'], 0)
        elif self.player_name == 'Defender':
            if places['aP1'] >= 1:
                _set_all_rates(rates, ['aT13', 'aT14'], 10)
                _set_all_rates(rates, ['aT12'], 0)
            elif places['bP1'] >= 1:
                _set_all_rates(rates, ['bT17'], 10)
                _set_all_rates(rates, ['bT15', 'bT16', 'bT18'], 0)
            elif places['cP1'] >= 1:
                _set_all_rates(rates, ['cT18'], 10)
                _set_all_rates(rates, ['cT19', 'cT20', 'cT21', 'cT22'], 0)
            elif places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)
            elif places['aP16'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT16'], 10)
                _set_all_rates(rates, ['bT15', 'bT17', 'bT18'], 0)
            elif places['aP16'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT18'], 10)
                _set_all_rates(rates, ['cT19', 'cT20', 'cT21', 'cT22'], 0)
            elif places['bP18'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT18'], 10)
                _set_all_rates(rates, ['cT19', 'cT20', 'cT21', 'cT22'], 0)
            elif places['aP16'] >= 1 and places['bP19'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT18', 'cT19', 'cT20'], 10)
                _set_all_rates(rates, ['cT21', 'cT22'], 0)
            elif places['aP16'] >= 1 and places['bP16'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT18', 'cT21', 'cT22'], 10)
                _set_all_rates(rates, ['cT19', 'cT20'], 0)
            elif places['aP14'] >= 1 and places['aP15'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT19', 'cT22'], 10)
                _set_all_rates(rates, ['cT18', 'cT20', 'cT21'], 0)

        return np.fromiter(rates.values(), dtype=float)


def _set_all_rates(rates, transitions, x):
    """
    Internal method used ot easily update transition rates
    :param rates: dictionary of rate values to update
    :param transitions: transitions to update
    :param x: value to update transitions to
    """
    for t in transitions:
        rates[t] = x
