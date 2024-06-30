import numpy as np


class PnpscNet():
    """
    Data class used to store all information associated with a PNPNSC net for all players
    """
    def __init__(self, json):
        """
        :param json: PNPSC net definition in json format
        """
        self.json = json
        self.done = False
        self.players = [item['name'] for item in json['players']]
        self.costs = {p: 0 for p in self.players}

        # Load the initial names and rates for transitions, ordered to allow for easy mapping later
        self.rates = {t['name']: t['rate'] for t in sorted(json['transitions'], key=lambda x: x['name'])}

        self.places = {p['name']: p['marking'] for p in sorted(json['places'], key=lambda x: x['name'])}

        self.controlled_rates = {}
        self.visible_places = {}
        for player in self.players:
            self.visible_places[player] = [p['name'] for p in sorted(json['places'], key=lambda x: x['name'])
                                           if
                                           p['player_observable'] is not None and player in p[
                                               'player_observable'].split(
                                               ',')]
            self.controlled_rates[player] = [t['name'] for t in sorted(json['transitions'], key=lambda x: x['name'])
                                             if
                                             t['player_control'] == player]

        self.goal_places = {p: [] for p in self.players}
        for p in json['places']:
            if 'goal' in p and p['goal'] in self.players:
                self.goal_places[p['goal']].append(p['name'])


    def get_all_places(self):
        """
        Returns all places for the PNPSC net
        :return: the current marking of the PNPSC net
        """
        return self.places

    def get_all_rates(self):
        """
        Returns all transition rates for the PNPSC net
        :return: a list of transition rates
        """
        return self.rates

    def get_visible_places(self, player_name):
        """
        Returns all places visible to a particular player
        :param player_name: name of the player
        :return: the current visible marking for a player
        """
        return {p: self.places[p] for p in self.visible_places[player_name]}

    def get_controlled_rates(self, player_name):
        """
        Returns all transition rates visible to a particular player
        :param player_name: name of the player
        :return: a list of transition rates visible to a player
        """
        return {r: self.rates[r] for r in self.controlled_rates[player_name]}

    def get_player_cost(self, player_name):
        """
        Returns the current accumulated cost for a player
        :param player_name: name of the player
        :return: the accumulated cost for a player
        """
        return self.costs[player_name]

    def get_json(self):
        """
        Returns the full json data used to initialize the PNPSC net
        :return: the PNPSC net in json format
        """
        return self.json

    def update_net(self, json):
        """
        Update the marking and rates of the PNPSC net with the provided json data
        :param json: json data from the simulator
        """
        self.done = json['end_of_run']
        for p in json['places']:
            self.places[p['name']] = p['marking']

        for p in json['players']:
            self.costs[p['name']] = p['cost']

    def get_marked_places(self):
        """
        Get a list of all marked places, ideal for printing
        :return: list of places that contain at least one token
        """
        return np.array(list(self.get_all_places().keys()))[
                  np.where(np.array(list(self.get_all_places().values())) > 0)]

    def get_goal_places(self, player_name):
        """
        Get goals of the specified player
        :return: list of goal places in the net
        """
        return self.goal_places[player_name]

    def get_end_places(self, player_name):
        """
        Get goals of all other players
        :return: list of ending places in the net
        """
        if len(self.players) < 2:
            return []
        return [value for key, value in self.goal_places.items() if key != player_name][0]
