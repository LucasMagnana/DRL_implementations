from random import sample
from random import * 
import torch.nn.functional as F

import numpy as np

import copy
import numpy as np
import torch
import torch.nn as nn

from python.NeuralNetworks import *
import matplotlib.pyplot as plt


class TD3Agent(object):
    def __init__(self, ob_space, ac_space, hyperParams, cuda=False, actor_to_load=None):

        self.hyperParams = hyperParams

        self.buffer_size = self.hyperParams.BUFFER_SIZE
        self.alpha = self.hyperParams.TAU
        self.gamma = self.hyperParams.GAMMA
        self.exploration_noise = self.hyperParams.EXPLORATION_NOISE
        self.policy_noise = self.hyperParams.POLICY_NOISE
        self.batch_size = self.hyperParams.BATCH_SIZE

        self.ac_space = ac_space

        self.tab_erreur = []

        self.noise = OUNoise(ac_space.shape[0])

        self.device = torch.device("cuda" if cuda else "cpu")

        self.critic_1 = Critic(ob_space.shape[0], ac_space.shape[0], self.hyperParams).to(device=self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1).to(device=self.device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), self.hyperParams.LR_CRITIC, weight_decay=self.hyperParams.WEIGHT_DECAY)
        
        self.critic_2 = Critic(ob_space.shape[0], ac_space.shape[0], self.hyperParams).to(device=self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2).to(device=self.device)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), self.hyperParams.LR_CRITIC, weight_decay=self.hyperParams.WEIGHT_DECAY)
        
        self.actor = Actor(ob_space.shape[0], ac_space.shape[0], self.hyperParams, max_action=ac_space.high[0], tanh=True).to(device=self.device)
        if(actor_to_load != None):
            self.actor.load_state_dict(torch.load(actor_to_load))
            self.actor.eval()
            self.noise = None
        
        self.actor_target = copy.deepcopy(self.actor).to(device=self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR_ACTOR)

        self.batch_prec_states = torch.zeros((self.hyperParams.BUFFER_SIZE,) + ob_space.shape)
        self.batch_actions = torch.zeros((self.hyperParams.BUFFER_SIZE,) + self.ac_space.shape)
        self.batch_states = torch.zeros((self.hyperParams.BUFFER_SIZE,) + ob_space.shape)
        self.batch_rewards = torch.zeros(self.hyperParams.BUFFER_SIZE)
        self.batch_dones = torch.zeros(self.hyperParams.BUFFER_SIZE)

        self.b_inds = np.arange(self.hyperParams.BUFFER_SIZE)

        self.num_transition_stored = 0
        


    def act(self, observation):
        action = self.actor(torch.tensor(observation,  dtype=torch.float32, device=self.device)).data.numpy()
        if(self.noise != None):
            action += self.noise.sample() #np.random.normal(0, self.exploration_noise, size=self.ac_space.shape[0])
        action = action.clip(self.ac_space.low, self.ac_space.high)
        return torch.tensor(action), []
        

    def memorize(self, ob_prec, action, ob, reward, done, infos):
        self.batch_prec_states[self.num_transition_stored%self.hyperParams.BUFFER_SIZE] = torch.Tensor(ob_prec)
        self.batch_actions[self.num_transition_stored%self.hyperParams.BUFFER_SIZE] = action
        self.batch_states[self.num_transition_stored%self.hyperParams.BUFFER_SIZE] = torch.Tensor(ob)
        self.batch_rewards[self.num_transition_stored%self.hyperParams.BUFFER_SIZE] = torch.tensor(reward).view(-1)
        self.batch_dones[self.num_transition_stored%self.hyperParams.BUFFER_SIZE] = not(done)

        self.num_transition_stored += 1

    def end_episode(self):
        self.noise.reset()

    def learn(self):
        
        for i in range(5):
            
            m_batch_inds = np.random.choice(self.b_inds[:min(self.num_transition_stored, self.hyperParams.BUFFER_SIZE)],\
            size=self.hyperParams.BATCH_SIZE, replace=False)
            
            tens_state = self.batch_prec_states[m_batch_inds]
            tens_state_next = self.batch_states[m_batch_inds]
            tens_action = self.batch_actions[m_batch_inds]
            tens_reward = self.batch_rewards[m_batch_inds].float()
            tens_done = self.batch_dones[m_batch_inds].bool()
            
            tens_noise = torch.empty(tens_action.shape)
            tens_noise = nn.init.normal_(tens_noise, mean=0, std=self.policy_noise)
            tens_noise = tens_noise.clamp(-self.hyperParams.NOISE_CLIP, self.hyperParams.NOISE_CLIP)
            tens_next_action = (self.actor_target(tens_state_next) + tens_noise)
            tens_next_action = tens_next_action.clamp(-self.ac_space.high[0], self.ac_space.high[0])

            tens_target_qvalue_1 = self.critic_1_target(tens_state_next, tens_next_action.float()).squeeze()
            tens_target_qvalue_2 = self.critic_2_target(tens_state_next, tens_next_action.float()).squeeze()       
            tens_target_qvalue = torch.min(tens_target_qvalue_1, tens_target_qvalue_2)
            
            tens_target_qvalue = tens_reward+(self.gamma*tens_target_qvalue)*tens_done.detach()


            tens_qvalue_1 = self.critic_1(tens_state, tens_action.float()).squeeze()
            critic_1_loss = F.mse_loss(tens_qvalue_1, tens_target_qvalue)
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward(retain_graph=True)
            self.critic_1_optimizer.step()

            tens_qvalue_2 = self.critic_2(tens_state, tens_action.float()).squeeze()
            critic_2_loss = F.mse_loss(tens_qvalue_2, tens_target_qvalue)
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()
            
            if(i%self.hyperParams.POLICY_DELAY == 0):
                
                actor_loss = -self.critic_1(tens_state, self.actor(tens_state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                    target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )

                for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
                    target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )




class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # Thanks to Hiu C. for this tip, this really helped get the learning up to the desired levels
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)

        self.state = x + dx
        return self.state
            

