import numpy as np

from .abstract_agent import AbstractAgent


class Capec66Agent(AbstractAgent):
    """
    A reimplementation of the optimal strategy for CAPEC-66 from [Bland, 2020]. This agent is
    only assumed to work for the exact CAPEC-66 net structure and rates found in the cited paper.
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
            if places['aP3'] >= 1:
                _set_all_rates(rates, ['aT6'], 10)
                _set_all_rates(rates, ['aT2'], 0)
            elif places['aP8'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT2', 'bT8', 'bT11'], 10)
                _set_all_rates(rates, ['bT5'], 10)
            elif places['aP5'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT5', 'bT8', 'bT11'], 10)
                _set_all_rates(rates, ['bT2'], 0)
            elif places['bP10'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT6', 'cT9', 'cT12'], 10)
                _set_all_rates(rates, ['cT3'], 0)
            elif places['bP7'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT6', 'cT9', 'cT12'], 10)
                _set_all_rates(rates, ['cT3'], 0)
            elif places['aP8'] >= 1 and places['bP3'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT3', 'cT9', 'cT12'], 10)
                _set_all_rates(rates, ['cT6'], 0)
            elif places['aP8'] >= 1 and places['bP13'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT3', 'cT9', 'cT12'], 10)
                _set_all_rates(rates, ['cT6'], 0)
            elif places['aP8'] >= 1 and places['bP13'] >= 1 and places['cP4'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT2', 'dT15', 'dT16'], 10)
                _set_all_rates(rates, ['dT6'], 0)
            elif places['aP5'] >= 1 and places['bP7'] >= 1 and places['cP8'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT15', 'dT16'], 10)
                _set_all_rates(rates, ['dT2', 'dT6'], 0)
            elif places['bP7'] >= 1 and places['cP8'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT2', 'dT6', 'dT15', 'dT16'], 10)
        elif self.player_name == 'Defender':
            #print(net.get_controlled_rates('Defender'))
            if places['aP1'] >= 1:
                _set_all_rates(rates, ['aT9', 'aT11'], 10)
            elif places['bP1'] >= 1:
                _set_all_rates(rates, ['bT15'], 10)
                _set_all_rates(rates, ['bT16', 'bT17', 'bT18'], 0)
            elif places['cP1'] >= 1:
                _set_all_rates(rates, ['cT15'], 10)
                _set_all_rates(rates, ['cT16', 'cT17', 'cT18'], 0)
            elif places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)
            elif places['aP10'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT16'], 10)
                _set_all_rates(rates, ['bT15', 'bT17', 'bT18'], 0)
            elif places['aP12'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT15'], 10)
                _set_all_rates(rates, ['bT16', 'bT17', 'bT18'], 0)
            elif places['aP12'] >= 1 and places['bP16'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT15', 'bT16'], 10)
                _set_all_rates(rates, ['bT17', 'bT18'], 0)
            elif places['aP12'] >= 1 and places['bP16'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT15', 'cT16'], 10)
                _set_all_rates(rates, ['cT17', 'cT18'], 0)
            elif places['aP12'] >= 1 and places['cP16'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT15'], 10)
                _set_all_rates(rates, ['cT16', 'cT17', 'cT18'], 0)
            elif places['cP16'] >= 1 and places['cP19'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT15', 'cT16', 'cT18'], 10)
                _set_all_rates(rates, ['cT17'], 0)

        return list(rates.values())


def _set_all_rates(rates, transitions, x):
    """
    Internal method used ot easily update transition rates
    :param rates: dictionary of rate values to update
    :param transitions: transitions to update
    :param x: value to update transitions to
    """
    for t in transitions:
        rates[t] = x
