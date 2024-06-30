import numpy as np

from .abstract_agent import AbstractAgent


class Capec163Agent(AbstractAgent):
    """
    A reimplementation of the optimal strategy for CAPEC-163 from [Bland, 2020]. This agent is
    only assumed to work for the exact CAPEC-163 net structure and rates found in the cited paper.
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
            if places['aP1'] >= 1:
                _set_all_rates(rates, ['aT1'], 10)
            elif places['aP4'] >= 1 and places['aP10'] >= 1 and places['aP16'] >= 1 and places['bP1'] >= 1:
                _set_all_rates(rates, ['bT3', 'bT9', 'bT16'], 10)
                _set_all_rates(rates, ['bT2', 'bT10', 'bT13', 'bT19'], 0)
            elif places['bP1'] >= 1:
                _set_all_rates(rates, ['bT3', 'bT9', 'bT10', 'bT13', 'bT16', 'bT19'], 10)
                _set_all_rates(rates, ['bT2'], 0)
            elif places['cP1'] >= 1:
                _set_all_rates(rates, ['cT3', 'cT6'], 10)
            elif places['aP4'] >= 1 and places['aP10'] >= 1 and places['bP4'] >= 1 and places['bP6'] >= 1 \
                    and places['bP12'] >= 1 and places['bP17'] >= 1 and places['cP1'] >= 1:
                _set_all_rates(rates, ['cT3'], 10)
                _set_all_rates(rates, ['cT6'], 0)
            elif places['aP4'] >= 1 and places['aP7'] >= 1 and places['aP13'] >= 1 and places['aP16'] >= 1 \
                    and places['bP4'] >= 1 and places['bP6'] >= 1 and places['bP17'] >= 1 and places['cP4'] >= 1 \
                    and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT2', 'dT16'], 10)
                _set_all_rates(rates, ['dT6', 'dT15'], 0)
            elif places['aP4'] >= 1 and places['aP10'] >= 1 and places['aP16'] >= 1 and places['aP19'] >= 1 \
                    and places['bP10'] >= 1 and places['bP17'] >= 1 and places['cP4'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT2', 'dT6'], 10)
                _set_all_rates(rates, ['dT15', 'dT16'], 0)
        elif self.player_name == 'Defender':
            if places['cP1'] >= 1:
                _set_all_rates(rates, ['cT9', 'cT10'], 10)
            elif places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)
            elif places['cP11'] >= 1 and places['dP9'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)
            elif places['cP10'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)
            elif places['cP10'] >= 1 and places['dP9'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)
            elif places['dP9'] >= 1 and places['dP1'] >= 1:
                _set_all_rates(rates, ['dT11'], 10)

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
