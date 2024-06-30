import numpy as np
from stable_baselines3.dqn import DQN

from .abstract_agent import AbstractAgent
from ..env.wrappers.discrete_pnpsc_wrapper import DiscretePnpscWrapper


def replace_rate(x, o):
    """
    Default update function to replace with the desired option
    """
    return o


class DqnAgent(AbstractAgent):
    """
    Deep Q-Learning agent implementation
    Each step the agent can perform from 1 to max_actions number of update to the controlled rates.
    The agent has a fixed function it can apply to any controlled rate with one of its optional values.
    """
    def __init__(self, env, f=replace_rate, options=(0, 10), max_actions=1, model_kwargs=None):
        """
        :param env: learning environment
        :param eval_env: optional evaluation environment
        :param f: update function for rates
        :param options: update options for f
        :param max_actions: maximum updates each simulator step
        :param model_kwargs: deviations from model default hyperparameters
        """
        super().__init__(env.player_name)
        self.env = DiscretePnpscWrapper(env, f, options, max_actions)
        self.policy_type = "MlpPolicy"
        self.max_actions = max_actions

        default_kwargs = dict(policy='MlpPolicy', env=self.env, gradient_steps=-1, tau=1, policy_kwargs=dict(net_arch=[128, 64]),
                                batch_size=4096, learning_rate=1e-5, gamma=0,
                                exploration_fraction=0.75, device='cpu', exploration_final_eps=0.05, learning_starts=10_000)

        if model_kwargs is not None:
            for k, v in model_kwargs.items():
                default_kwargs[k] = v

        self.model = DQN(**default_kwargs)

    def load_model(self, file_path):
        """
        Load an exisiting model from file. This will replace the current model
        :param file_path: path to saved model
        """
        self.model = DQN.load(file_path, self.env)

    def _act(self, net, print_strategy=False):
        """
        Performs the player action for the given state of the net
        :param net: the current pnpsc net object
        :param print_strategy: toggle to output the strategy of the player
        :return: The desired rates for the player controlled transitions
        """
        places = list(net.get_visible_places(self.player_name).values())
        rates = net.get_controlled_rates(self.player_name)

        for i in range(self.max_actions):
            state = np.concatenate([places, list(rates.values()), [i]], dtype=np.float32)

            action, _ = self.model.predict(state, deterministic=print_strategy)
            new_rates = self.env.generate_action(action, rates)
            if new_rates is None:
                break
            rates = new_rates

        return list(rates.values())



