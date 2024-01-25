
import random

from torch.nn import MSELoss, HuberLoss
from collections import OrderedDict

import numpy as np

import copy
import numpy as np
import torch

from python.NeuralNetworks import *

class DQNAgent(object):
    def __init__(self, ob_space, ac_space, hyperParams, test=False, double=True, duelling=True, PER=False, cuda=False, actor_to_load=None, cnn=False):

        self.hyperParams = hyperParams

        self.ac_space = ac_space 

        self.epsilon = self.hyperParams.START_EPSILON
        self.epsilon_decay_value = self.hyperParams.FIRST_EPSILON_DECAY
        
        self.gamma = self.hyperParams.GAMMA
        self.test = False

        self.device = torch.device("cuda" if cuda else "cpu")

        self.duelling = duelling
        self.cnn = cnn
        if(self.duelling):
            self.actor = ActorCritic(ob_space.shape[0], ac_space.n, self.hyperParams, cnn=cnn).to(self.device) 
        else:
            self.actor = Actor(ob_space.shape[0], ac_space.n, self.hyperParams, cnn=cnn).to(self.device) #for cartpole


        if(actor_to_load != None): #if it's a test, use the loaded NN
            self.epsilon = 0.01
            actor = torch.load(actor_to_load, map_location=self.device)
            self.actor.load_state_dict(actor)
            self.actor.eval()
            self.test = True
        else:
            self.actor_target = copy.deepcopy(self.actor) #a target network is used to make the convergence possible (see papers on DRL)

            self.optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR) # smooth gradient descent

            self.ob_space = ob_space

            self.double = double

            self.update_target = 0

            self.tab_max_q = []
            self.tab_loss = []

            self.buffer_size = int(self.hyperParams.BUFFER_SIZE)

            if(cnn):
                ob_dtype = torch.uint8
            else:
                ob_dtype = torch.float

            self.batch_prec_states = torch.zeros((self.buffer_size,) + self.ob_space.shape, dtype=ob_dtype)
            self.batch_actions = torch.zeros((self.buffer_size,) + self.ac_space.shape)
            self.batch_states = torch.zeros((self.buffer_size,) + self.ob_space.shape, dtype=ob_dtype)
            self.batch_rewards = torch.zeros(self.buffer_size)
            self.batch_dones = torch.zeros(self.buffer_size)

            self.b_inds = np.arange(self.buffer_size)

            self.num_transition_stored = 0

            self.loss = HuberLoss()



    def epsilon_decay(self):
        self.epsilon -= self.epsilon_decay_value
        if(self.epsilon_decay_value == self.hyperParams.FIRST_EPSILON_DECAY and self.epsilon<self.hyperParams.FIRST_MIN_EPSILON ):
            self.epsilon = self.hyperParams.FIRST_MIN_EPSILON
            self.epsilon_decay_value = self.hyperParams.SECOND_EPSILON_DECAY
        if(self.epsilon < self.hyperParams.MIN_EPSILON):
            self.epsilon = self.hyperParams.MIN_EPSILON

  
        


    def act(self, observation):
        if(not self.test):
            if(self.num_transition_stored >= self.hyperParams.LEARNING_START):
                self.epsilon_decay()
            self.update_target += 1  
            observation = np.expand_dims(observation, axis=0)
        

        observation = torch.tensor(observation, device=self.device)
        tens_qvalue = self.actor(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random.random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            action = indices #return it
        else:
            action = torch.tensor(random.randint(0, tens_qvalue.size()[0]-1)) #choose a random action

        return action, []


    def memorize(self, ob_prec, action, ob, reward, done, infos):
        self.batch_prec_states[self.num_transition_stored%self.buffer_size] = torch.Tensor(np.array(ob_prec))
        self.batch_actions[self.num_transition_stored%self.buffer_size] = action
        self.batch_states[self.num_transition_stored%self.buffer_size] = torch.Tensor(np.array(ob))
        self.batch_rewards[self.num_transition_stored%self.buffer_size] = torch.tensor(reward).view(-1)
        self.batch_dones[self.num_transition_stored%self.buffer_size] = not(done)

        self.num_transition_stored += 1
  

    def learn(self, n_iter=None):


        m_batch_inds = np.random.choice(self.b_inds[:min(self.num_transition_stored, self.buffer_size)],\
        size=self.hyperParams.BATCH_SIZE, replace=False)
        
        tens_state = self.batch_prec_states[m_batch_inds]
        tens_state_next = self.batch_states[m_batch_inds]
        tens_action = self.batch_actions[m_batch_inds].long()
        tens_reward = self.batch_rewards[m_batch_inds].float()
        tens_done = self.batch_dones[m_batch_inds].bool()

        tens_qvalue = self.actor(tens_state) #compute the qvalues for all the actual states

        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag() #select the qvalues corresponding to the chosen actions


        
        if(self.double):
            # Double DQN
            tens_next_qvalue = self.actor(tens_state_next) #compute all the qvalues for all the "next states" with the ppal network
            (_, tens_next_action) = torch.max(tens_next_qvalue, 1) #returns the indices of the max qvalues for all the next states(to choose the next actions)
            tens_next_qvalue = self.actor_target(tens_state_next) #compute all the qvalues for all the "next states" with the target network
            tens_next_qvalue = torch.index_select(tens_next_qvalue, 1, tens_next_action).diag() #select the qvalues corresponding to the chosen next actions          
        else:
            # Simple DQN
            tens_next_qvalue = self.actor_target(tens_state_next) #compute all the qvalues for all the "next states"
            (tens_next_qvalue, _) = torch.max(tens_next_qvalue, 1) #select the max qvalues for all the next states
            

        self.optimizer.zero_grad() #reset the gradient
        tens_loss = self.loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient

        self.tab_max_q.append(torch.max(tens_qvalue).item())
        self.tab_loss.append(torch.max(tens_loss).item())

        #print(tens_loss.item(), torch.max(tens_qvalue))
        
        if(self.update_target/self.hyperParams.TARGET_UPDATE > 1):
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.update_target = 0