import argparse
import sys
import gym 
import pickle
import copy
import torch
import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from python.NeuralNetworks import PPO_Actor, PPO_Critic
from python.hyperParams import PPOHyperParams, module

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


class PPOAgent():

    def __init__(self, hyperParams, ob_space, ac_space, actor_to_load, critic_to_load):

        self.hyperParams = hyperParams
        self.old_actor = PPO_Actor(env.observation_space.shape[0], env.action_space.n, hyperParams)
        self.critic = PPO_Critic(env.observation_space.shape[0], hyperParams)

        if(actor_to_load != None and critic_to_load != None):
            self.old_actor.load_state_dict(torch.load(actor_to_load))
            self.old_actor.eval()
            self.critic.load_state_dict(torch.load(critic_to_load))
            self.critic.eval()
        else:
            self.actor = copy.deepcopy(self.old_actor)   
            # Define optimizer
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.hyperParams.LR)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.hyperParams.LR)
            self.mse = torch.nn.MSELoss()

        self.avg_rewards = []
        self.total_rewards = []

        self.ep = 0

        self.reset_batches()


    def reset_batches(self):
        self.batch_rewards = []
        self.batch_advantages = []
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_done = []


    def learn(self):

        for k in range(self.hyperParams.K):
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            state_tensor = torch.tensor(self.batch_states)
            advantages_tensor = torch.tensor(self.batch_advantages)

            values_tensor = torch.tensor(self.batch_values)
            rewards_tensor = torch.tensor(self.batch_rewards, requires_grad = True)
            rewards_tensor = rewards_tensor.float()
            # Actions are used as indices, must be 
            # LongTensor
            action_tensor = torch.LongTensor(self.batch_actions)
            action_tensor = action_tensor.long()

            
            # Calculate actor loss
            probs = self.actor(state_tensor)
            selected_probs = torch.index_select(probs, 1, action_tensor).diag()

            old_probs = self.old_actor(state_tensor)
            selected_old_probs = torch.index_select(old_probs, 1, action_tensor).diag()

            loss = selected_probs/selected_old_probs*advantages_tensor
            clipped_loss = torch.clamp(selected_probs/selected_old_probs, 1-self.hyperParams.EPSILON, 1+self.hyperParams.EPSILON)*advantages_tensor

            loss_actor = -torch.min(loss, clipped_loss).mean()



            # Calculate gradients
            loss_actor.backward()
            # Apply gradients
            self.optimizer_actor.step()

            # Calculate critic loss
            loss_critic = self.mse(values_tensor, rewards_tensor)  
            # Calculate gradients
            loss_critic.backward()
            # Apply gradients
            self.optimizer_critic.step()

        self.old_actor = copy.deepcopy(self.actor)


    def act(self, env, render=False):

        for y in range(self.hyperParams.NUM_EP_ENV):
            ob_prec = env.reset()
            states = []
            rewards = []
            actions = []
            values = []
            list_done = []
            done = False
            step=1
            while(not done and step<self.hyperParams.MAX_STEPS):
                if(render):
                    env.render()
                # Get actions and convert to numpy array
                action_probs = self.old_actor(torch.tensor(ob_prec)).detach().numpy()
                val = self.critic(torch.tensor(ob_prec)).detach().numpy()
                action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
                ob, r, done, _ = env.step(action)

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
            ar = np.mean(self.total_rewards[-100:])
            self.avg_rewards.append(ar)

            self.ep += 1

            if(not render):
                print("\rEp: {} Average of last 100: {:.2f}".format(self.ep, ar), end="")




if __name__ == '__main__':
    env = gym.make(module) #gym env

    list_envs = []
    testing = False

    hyperParams = PPOHyperParams()

    actor_to_load=None
    critic_to_load=None

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "--test"):
            testing = True
            with open('./trained_networks/'+module+'_PPO.hp', 'rb') as infile:
                hyperParams = pickle.load(infile)

            actor_to_load='./trained_networks/'+module+'_ac_PPO.n'
            critic_to_load='./trained_networks/'+module+'_cr_PPO.n'

            hyperParams.EPISODE_COUNT=1
            hyperParams.NUM_AGENTS=1
            hyperParams.K=0
            hyperParams.NUM_EP_ENV=1

    ppo_agent = PPOAgent(hyperParams, env.observation_space.shape[0], env.action_space.n, actor_to_load, critic_to_load)

    for ep in range(hyperParams.EPISODE_COUNT):
        # Set up lists to hold results
        ppo_agent.reset_batches()


        for a in range(hyperParams.NUM_AGENTS):
            if(len(list_envs) < hyperParams.NUM_AGENTS):
                list_envs.append(gym.make(module))

            ppo_agent.act(env, render=testing)

        if(not testing):
            ppo_agent.learn()
            
    if(not testing):
        print()
        
    env.close()

    if(testing):
        print("Sum of rewards:", ppo_agent.total_rewards[-1])
    else:
        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure(figsize=(25, 12), dpi=80)
        plt.plot(ppo_agent.total_rewards, linewidth=1)
        plt.plot(ppo_agent.avg_rewards, linewidth=1)
        plt.ylabel('Sum of the rewards')       
        plt.savefig("./images/"+module+"_PPO.png")
        
        #save the neural networks of the policy
        torch.save(ppo_agent.old_actor.state_dict(), './trained_networks/'+module+'_ac_PPO.n')
        torch.save(ppo_agent.critic.state_dict(), './trained_networks/'+module+'_cr_PPO.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/'+module+'_PPO.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
                

                