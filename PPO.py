import argparse
import sys
import gym 
import pickle

import threading

import matplotlib.pyplot as plt

import datetime as dt

from python.NeuralNetworks import PPO_Actor, PPO_Critic
from python.TD3Agent import *
from python.DDQNAgent import *
from python.hyperParams import PPOHyperParams, module

from test import test


class MonThread (threading.Thread):
    def __init__(self, module, hyperParams, actor, critic):
        threading.Thread.__init__(self)

        self.env = gym.make(module)
        self.hyperParams = hyperParams

        self.batch_rewards = []
        self.batch_advantages = []
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_done = []
        self.total_rewards = []

        self.actor = actor
        self.critic = critic

    def run(self):
        for _ in range(self.hyperParams.NUM_EP_ENV):
            ob_prec = self.env.reset()
            states = []
            rewards = []
            actions = []
            values = []
            list_done = []
            done = False
            step=1
            while(not done and step<hyperParams.MAX_STEPS):
                # Get actions and convert to numpy array
                action_probs = self.actor(torch.tensor(ob_prec)).detach().numpy()
                val = self.critic(torch.tensor(ob_prec)).detach().numpy()
                action = np.random.choice(action_space, p=action_probs)
                ob, r, done, _ = self.env.step(action)

                states.append(ob_prec)
                values.extend(val)
                rewards.append(r)
                actions.append(action)
                list_done.append(done)  

                ob_prec = ob
                step+=1                

            self.batch_rewards.extend(discount_rewards(rewards, hyperParams.GAMMA))
            gaes = gae(np.expand_dims(np.array(rewards), 0), np.expand_dims(np.array(values), 0), np.expand_dims(np.array([not elem for elem in list_done]), 0), hyperParams.GAMMA, hyperParams.LAMBDA)
            self.batch_advantages.extend(gaes[0])
            self.batch_states.extend(states)
            self.batch_values.extend(values)
            self.batch_actions.extend(actions)
            self.batch_done.extend(list_done)
            self.total_rewards.append(sum(rewards))


def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def gae(rewards, values, episode_ends, gamma, lam):
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

    hyperParams = PPOHyperParams()

    actor = PPO_Actor(env.observation_space.shape[0], env.action_space.n, hyperParams)
    old_actor = copy.deepcopy(actor)
    critic = PPO_Critic(env.observation_space.shape[0], hyperParams)

    
    # Define optimizer
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=hyperParams.LR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=hyperParams.LR)

    mse = torch.nn.MSELoss()
    
    action_space = np.arange(env.action_space.n)

    total_rewards = []

    for ep in range(hyperParams.EPISODE_COUNT):
        # Set up lists to hold results
        batch_advantages = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_done = []
        batch_values = []
        threads=[]
        for a in range(hyperParams.NUM_AGENTS):
            threads.append(MonThread(module, hyperParams, batch_rewards, batch_advantages, batch_states, batch_values, batch_actions, batch_done, total_rewards, copy.deepcopy(old_actor), copy.deepcopy(critic)))
            threads[-1].start()

        print(batch_states)

        for k in range(hyperParams.K):
                
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
            print(state_tensor.shape)
            probs = actor(state_tensor)
            selected_probs = torch.index_select(probs, 1, action_tensor).diag()

            old_probs = old_actor(state_tensor)
            selected_old_probs = torch.index_select(old_probs, 1, action_tensor).diag()

            loss = selected_probs/selected_old_probs*advantages_tensor
            clipped_loss = torch.clamp(selected_probs/selected_old_probs, 1-hyperParams.EPSILON, 1+hyperParams.EPSILON)*advantages_tensor

            loss_actor = -torch.min(loss, clipped_loss).mean()


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

        old_actor = copy.deepcopy(actor)
                    
        avg_rewards = np.mean(total_rewards[-100:])
        # Print running average
        print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, avg_rewards), end="")
    print()

                