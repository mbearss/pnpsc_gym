from .abstract_agent import AbstractAgent


class StaticAgent(AbstractAgent):
    """
    A PNPSC player agent that takes no action
    """
    def __init__(self, player_name):
        """
        :param player_name: Name of the player, must match a player listed in the net definition
        """
        super().__init__(player_name)

    def _act(self, net, print_strategy=False):
        """
        Performs the player action for the given state of the net
        :param net: the current pnpsc net object
        :param print_strategy: toggle to output the strategy of the player
        :return: The desired rates for the player controlled transitions
        """
        return list(net.get_controlled_rates(self.player_name).values())

