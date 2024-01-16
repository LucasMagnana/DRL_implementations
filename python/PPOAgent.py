import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

import pickle


from python.NeuralNetworks import *

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor.mean().item())
                print(' - gradient:', tensor.grad.mean().item())
                print()
            except AttributeError as e:
                print()
                getBack(n[0])


def gae(rewards, values, dones, num_steps, nextdones, nextob, actor, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - torch.tensor(nextdones).float()
            with torch.no_grad():
                _, nextvalues = actor(torch.tensor(nextob))
                nextvalues = nextvalues.flatten()
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages




class PPOAgent():

    def __init__(self, ob_space, ac_space, hyperParams, actor_to_load=None, continuous_action_space=False, cnn=False):

        self.hyperParams = hyperParams

        self.continuous_action_space = continuous_action_space

        self.ac_space = ac_space
        self.ob_space = ob_space

        device = torch.device("cpu")

        if(self.continuous_action_space):
            self.actor = PPO_Actor(ob_space.shape[0], ac_space.shape[0], hyperParams, max_action=ac_space.high[0])
        else:  
            self.actor = ActorCritic(ob_space.shape, ac_space.n, self.hyperParams, cnn=cnn, ppo=True).to(device)

        if(actor_to_load != None):
            self.actor.load_state_dict(torch.load(actor_to_load))
            self.actor.eval()


        # Define optimizer
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hyperParams.LR, eps=1e-5)
        self.mse = torch.nn.MSELoss()

        

        if(self.continuous_action_space):
            self.action_std = torch.full((ac_space.n,), 1/100)

        self.num_decisions_made = 0

        self.e_loss = []
        self.p_loss = []
        self.v_loss = []
        self.lr = []


        self.batch_rewards = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_states = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV) + self.ob_space.shape)
        self.batch_values = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_actions = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV) + self.ac_space.shape)
        self.batch_log_probs = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
        self.batch_dones = torch.zeros((self.hyperParams.BATCH_SIZE, self.hyperParams.NUM_ENV))
    



    def memorize(self, ob_prec, action, ob, reward, done, infos):
        val = infos[0]
        action_probs = infos[1]
        env = infos[2]
        step = infos[3]
        self.batch_states[step] = torch.Tensor(ob_prec)
        self.batch_values[step] = val
        self.batch_rewards[step] = torch.tensor(reward).view(-1)
        self.batch_actions[step] = action
        self.batch_log_probs[step] = action_probs
        if(self.hyperParams.NUM_ENV == 1):
            done = [done]
        self.batch_dones[step] = torch.Tensor(done)


    def act(self, observation):
        with torch.no_grad():
            if(self.continuous_action_space):
                action_exp, action_std = self.actor(torch.tensor(observation))
                std_mat = torch.diag(action_std)
                dist = MultivariateNormal(action_exp, std_mat)
            else:
                action_probs, val = self.actor(torch.tensor(observation))
                dist = Categorical(logits=action_probs)

            action = dist.sample()
            self.num_decisions_made += 1

            return action, [val.flatten(), dist.log_prob(action)]



    
    def learn(self, next_dones, next_obs):
        gaes = gae(self.batch_rewards, self.batch_values, self.batch_dones, self.hyperParams.BATCH_SIZE, next_dones, next_obs, self.actor, self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
        returns = gaes + self.batch_values
        # flatten the batch
        batch_states = self.batch_states.reshape((-1,) + self.ob_space.shape)
        batch_logprobs = self.batch_log_probs.reshape(-1)
        batch_actions = self.batch_actions.reshape((-1,) + self.ac_space.shape)
        batch_advantages = gaes.reshape(-1)
        batch_returns = returns.reshape(-1)
        #print(batch_returns.shape, batch_returns.mean().item())
        batch_values = self.batch_values.reshape(-1)


        # Optimizing the policy and value network
        batch_inds = np.arange(len(batch_states))
        clipfracs = []
        for epoch in range(self.hyperParams.K):
            np.random.shuffle(batch_inds)
            for start in range(0, len(batch_states), len(batch_states)//self.hyperParams.NUM_MINIBATCHES):                
                end = start + len(batch_states)//self.hyperParams.NUM_MINIBATCHES
                m_batch_inds = batch_inds[start:end]
                action_probs, newvalue = self.actor(batch_states[m_batch_inds])
                probs = Categorical(logits=action_probs)
                newlogprob = probs.log_prob(batch_actions.long()[m_batch_inds])
                entropy = probs.entropy()
                logratio = newlogprob - batch_logprobs[m_batch_inds]
                ratio = logratio.exp()
                norm_batch_advantages = batch_advantages[m_batch_inds]
                norm_batch_advantages = (norm_batch_advantages - norm_batch_advantages.mean()) / (norm_batch_advantages.std() + 1e-8)
        
                # Policy loss
                unclipped_policy_loss = -norm_batch_advantages * ratio
                clipped_policy_loss = -norm_batch_advantages * torch.clamp(ratio, 1 - self.hyperParams.EPSILON, 1 + self.hyperParams.EPSILON)
                e = unclipped_policy_loss
                policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss)
                policy_loss = policy_loss.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                value_loss_unclipped = (newvalue - batch_returns[m_batch_inds]) ** 2
                values_clipped = batch_values[m_batch_inds] + torch.clamp(
                    newvalue - batch_values[m_batch_inds],
                    -self.hyperParams.EPSILON,
                    self.hyperParams.EPSILON,
                )
                value_loss_clipped = (values_clipped - batch_returns[m_batch_inds]) ** 2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = 0.5 * value_loss_max.mean()
    

                entropy_loss = entropy.mean()

                loss = policy_loss - self.hyperParams.ENTROPY_COEFF * entropy_loss + value_loss * self.hyperParams.VALUES_COEFF


                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperParams.MAX_GRAD)
                #getBack(loss.grad_fn)
                self.optimizer.step()

        self.p_loss.append(policy_loss.item())
        self.v_loss.append(value_loss.item())
        self.e_loss.append(entropy_loss.item())


    '''def learn(self, next_dones, next_obs):
        gaes = gae(self.batch_rewards, self.batch_values, self.batch_dones, self.hyperParams.BATCH_SIZE, next_dones, next_obs, self.actor, self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
        returns = gaes + self.batch_values
                
        batch_states = self.batch_states.reshape((-1,)+self.ob_space.shape)
        batch_log_probs = self.batch_log_probs.reshape(-1)
        batch_actions = self.batch_actions.reshape((-1,)+self.ac_space.shape)
        batch_advantages = gaes.reshape(-1)
        batch_returns = returns.reshape(-1)

        batch_values = self.batch_values.reshape(-1)

        batch_inds = np.arange(len(batch_states))
        for k in range(self.hyperParams.K):
            np.random.shuffle(batch_inds)
            for start in range(0, len(batch_states), len(batch_states)//self.hyperParams.NUM_MINIBATCHES):
                end = start + len(batch_states)//self.hyperParams.NUM_MINIBATCHES
                m_batch_inds = batch_inds[start:end]
                
                if(self.continuous_action_space):
                    action_exp, action_std = self.actor(batch_states[m_batch_inds])
                    mat_std = torch.diag_embed(action_std)
                    dist = MultivariateNormal(action_exp, mat_std)

                else:
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_probs, new_values = self.actor(batch_states[m_batch_inds])
                    dist = Categorical(logits=action_probs)

                entropy = dist.entropy()
                new_log_probs = dist.log_prob(batch_actions.long()[m_batch_inds])
                logratios = new_log_probs - batch_log_probs[m_batch_inds]
                ratios = logratios.exp()
                
                #batch_advantages = batch_rewards - new_values.detach()  
                norm_batch_advantages = batch_advantages[m_batch_inds]
                norm_batch_advantages = (norm_batch_advantages-norm_batch_advantages.mean()) / (norm_batch_advantages.std() + 1e-8)
                unclipped_loss_actor = -norm_batch_advantages*ratios
                clipped_loss_actor = -norm_batch_advantages*torch.clamp(ratios, 1-self.hyperParams.EPSILON, 1+self.hyperParams.EPSILON)
                loss_actor = torch.max(unclipped_loss_actor, clipped_loss_actor)
                loss_actor = loss_actor.mean()

                new_values = new_values.view(-1)
                value_loss_unclipped = (new_values - batch_returns[m_batch_inds])**2

                values_clipped = batch_values[m_batch_inds] + torch.clamp(new_values-batch_values[m_batch_inds], -self.hyperParams.EPSILON, self.hyperParams.EPSILON)
                value_loss_clipped = (values_clipped - batch_returns[m_batch_inds])**2

                loss_critic_max = torch.max(value_loss_unclipped, value_loss_clipped)
                loss_critic = 0.5*loss_critic_max.mean()

                #loss_critic = self.mse(new_values, batch_rewards)

                loss_entropy = entropy.mean()
        

                loss = loss_actor - self.hyperParams.ENTROPY_COEFF * loss_entropy + loss_critic * self.hyperParams.VALUES_COEFF

                print(f'loss : {loss.item():.30f}')
                #print("vl:", loss_critic.item(), "pl:", loss_actor.item(), "el:", loss_entropy.item(), "l:", loss.item())
                print(loss_actor, loss_actor.item(), loss_actor.dtype)
                print()
                print("pl:", loss_actor.item(), "l:", loss.item(), self.actor.actor.weight.mean().item())
                # Reset gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperParams.MAX_GRAD)
                getBack(loss.grad_fn)
                # Apply gradients
                self.optimizer.step()
                print(zfes)


        self.p_loss.append(loss_actor.item())
        self.value_loss.append(loss_critic.item())
        self.e_loss.append(loss_entropy.item())'''