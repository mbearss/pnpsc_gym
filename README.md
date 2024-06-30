# PNPSC  Gymnasium

This repository contains code developed as part of my dissertation on reinforcement learning. The repository is structured to facilitate the use and development of PNPSC player agents within a custom gym environment.

## Repository Structure

- **nets/**: Contains JSON files defining PNPSC nets. These files describe the structure and properties of the nets used in the environment. 
- **src/pnpsc_env/agents/**: Contains several example agents that can be used within the gym environment. These agents demonstrate various approaches to solving tasks in the environment.
- **src/pnpsc_env/env/wrappers**: Contains wrappers designed to manipulate the gym environment for easier agent development. These wrappers help simplify interactions with the environment, making it more convenient to implement and test new agents.

## Installation

To get started with this RL gym environment, clone the repository and install the required dependencies:
```
pip install -r requirements.txt
```

## Example Agent Training

The following code demonstrates how to train the included DQN agent on the included example PNPSC net.
```
import numpy as np

from src.pnpsc_env.agents.dqn_agent import DqnAgent
from src.pnpsc_env.agents.random_agent import RandomAgent
from src.pnpsc_env.env.pnpsc_local_env import PnpscLocalEnv

if __name__ == '__main__':
    def evalAgent(env, agent):
        rewards = []
        for i in range(10000):
            env.reset()
            done, reward = False, 0
            while not done:
                state, r, done, _ = env.step(agent.act(env.net)[0])
                reward += r
            rewards.append(reward)

        print(np.mean(rewards))

    env = PnpscLocalEnv('Attacker', 'example_net.json', max_tokens=1)
    random = RandomAgent('Attacker')
    attacker = DqnAgent(env)

    evalAgent(env, random)
    evalAgent(env, attacker)
    attacker.model.learn(total_timesteps=100_000)
    evalAgent(env, attacker)

```
The agent is evaluated 10,000 times to ensure an accurate score. The score of the attacker agent should increase after the training is complete.

## Citation

If you use this code in your research, please cite my dissertation:

Bearss, Edwin Michael, "Extending machine learning of cyberattack strategies with continuous transition rates" (2023). _Dissertations_. 348.  
https://louis.uah.edu/uah-dissertations/348

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This project is part of my dissertation work on reinforcement learning. Special thanks to my advisors, colleagues, and the open-source community for their support and contributions.
