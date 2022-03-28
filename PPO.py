import argparse
import sys
import gym 
import pickle

import matplotlib.pyplot as plt

import datetime as dt

from python.NeuralNetworks import PPO_Actor, PPO_Critic
from python.TD3Agent import *
from python.DDQNAgent import *
from python.hyperParams import hyperParams, module

from test import test


def discount_rewards(rewards, gamma=hyperParams.GAMMA):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def gae(rewards, values, episode_ends, gamma=hyperParams.GAMMA, lam=0.5):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages



if __name__ == '__main__':
    env = gym.make(module) #gym env

    actor = PPO_Actor(env.observation_space.shape[0], env.action_space.n)
    critic = PPO_Critic(env.observation_space.shape[0])
    
    # Set up lists to hold results
    total_rewards = []
    batch_advantages = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_done = []
    batch_values = []
    batch_counter = 1
    
    # Define optimizer
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=hyperParams.LR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=hyperParams.LR)

    mse = torch.nn.MSELoss()
    
    action_space = np.arange(env.action_space.n)
    ep = 0
    while ep < hyperParams.EPISODE_COUNT:
        ob_prec = env.reset()
        states = []
        rewards = []
        actions = []
        values = []
        list_done = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = actor(torch.tensor(ob_prec)).detach().numpy()
            val = critic(torch.tensor(ob_prec)).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            ob, r, done, _ = env.step(action)

            states.append(ob_prec)
            values.extend(val)
            rewards.append(r)
            actions.append(action)
            list_done.append(done)

            ob_prec = ob
            
            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards))
                gaes = gae(np.expand_dims(np.array(rewards), 0), np.expand_dims(np.array(values), 0), np.expand_dims(np.array([not elem for elem in list_done]), 0))
                batch_advantages.extend(gaes[0])
                batch_states.extend(states)
                batch_values.extend(values)
                batch_actions.extend(actions)
                batch_done.extend(list_done)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == hyperParams.BATCH_SIZE:
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()

                    state_tensor = torch.tensor(batch_states)
                    advantages_tensor = torch.tensor(batch_advantages)

                    values_tensor = torch.tensor(batch_values)
                    rewards_tensor = torch.tensor(batch_rewards, requires_grad = True)
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    action_tensor = action_tensor.long()
                    
                    # Calculate actor loss
                    logprob = torch.log(actor(state_tensor))
                    selected_logprobs = advantages_tensor * torch.index_select(logprob, 1, action_tensor).diag()
                    loss_actor = -selected_logprobs.mean()                    
                    # Calculate gradients
                    loss_actor.backward()
                    # Apply gradients
                    optimizer_actor.step()

                    # Calculate critic loss
                    loss_critic = mse(values_tensor, rewards_tensor)  
                    # Calculate gradients
                    loss_critic.backward()
                    # Apply gradients
                    optimizer_critic.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_values = []
                    batch_advantages = []
                    batch_counter = 1
                    
                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, avg_rewards), end="")
                ep += 1
                