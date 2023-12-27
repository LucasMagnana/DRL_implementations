# DRL_implementations

This project contains the implementation of various deep reinforcement learning algorithms. Four algorithms have yet been implemented :

- REINFORCE [^1]
- Double Duelling Deep Q-Network (3DQN) [^2] [^3] [^4]
- Proximal Policy Optimization (PPO) [^5]
- Twin Delayed Deep Deterministic Policy Gradient (TD3) [^6]

## Trained Environments

Here are some [Gym](https://www.gymlibrary.dev/index.html) environments that have been solved (or nearly solved) using the implemented algorithms.
### Classic Control
| <img src="https://github.com/LucasMagnana/DRL_implementations/images/Acrobot-v1_PPO.gif?raw=true" width=300> | <img src="https://github.com/LucasMagnana/DRL_implementations/images/CartPole-v1_REINFORCE.gif?raw=true" width=300> | 
|:--:|:--:| 
| *Acrobot-v1 with PPO* | *CartPole-v1 with REINFORCE* |

### Box2D with Continuous Action Space
| <img src="https://github.com/LucasMagnana/DRL_implementations/images/LunarLanderContinuous-v2_TD3.gif?raw=true" width=300> | <img src="https://github.com/LucasMagnana/DRL_implementations/images/BipedalWalker-v3_TD3.gif?raw=true" width=300> | 
|:--:|:--:| 
| *LunarLanderContinuous-v2 with TD3* | *BipedalWalker-v3 with TD3* |

### Atari (using pixels)
| <img src="https://github.com/LucasMagnana/DRL_implementations/images/Pong-v5_3DQN.gif?raw=true" width=300> | <img src="https://github.com/LucasMagnana/DRL_implementations/images/Breakout-v5_3DQN.gif?raw=true" width=300> | 
|:--:|:--:| 
| *ALE/Pong-v5 with 3DQN* | *ALE/Breakout-v5 with 3DQN* |



## Dependencies and installation

This project uses [Python 3.10.12](https://www.python.org/downloads/release/python-31012/). Use the package manager [pip](https://pypi.org/project/pip/) to install the dependencies :

```bash
pip install -r requirements.txt
```

## Usage

The files that can be executed are `train.py`, `test.py` and `mp4_to_gif.py`. `train.py` trains a DRL agent using a specified algorithm on a sepcified [Gym](https://www.gymlibrary.dev/index.html) environment. The trained neural network used by the agent are saved in the `files/` directory. An image displaying the sum of rewards for each episode as well as the average sum of rewards for the last 100 episodes is also saved in the `images/` directory. The parameters of `train.py` are :

1. `-a`/`--algorithm` : specify the algorithm to use. Accepted string are `3DQN`, `PPO`, and `TD3`.

2. `-m`/`--module` : the Gym environment to train the algorithm on.

`test.py` displays an episode of a specified environment using an agent previously trained with a specified algorithm. `-a` and `-m` can be used with the same purposes as for `train.py`. If the `--save` parameter is used, the episode is not displayed but saved in the `rgb_array/` directory instead. `mp4_to_gif.py` can then be used to convert the video in a gif file and to save it in the `images/` directory. The `-a` and `-m` parameters are used by `mp4_to_gif.py` to name the GIF file.

> [!WARNING]
> PPO only works with classic control environments with discrete action space for now.

## References

[^1]: Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." *Advances in neural information processing systems* 12 (1999).
[^2]: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." *nature* 518.7540 (2015): 529-533.
[^3]: Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." *Proceedings of the AAAI conference on artificial intelligence.* Vol. 30. No. 1. 2016.
[^4]: Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." *International conference on machine learning.* PMLR, 2016.
[^5]: Schulman, John, et al. "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347* (2017).
[^6]: Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." *International conference on machine learning.* PMLR, 2018.



