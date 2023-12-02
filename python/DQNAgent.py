
import random
from random import sample, random, randint

from torch.nn import MSELoss, HuberLoss
from collections import deque

import numpy as np

import copy
import numpy as np
import torch

from python.NeuralNetworks import *

class DQNAgent(object):
    def __init__(self, observation_space, action_space, hyperParams, test=False, double=True, duelling=True, PER=False, cuda=False, actor_to_load=None, cnn=False):

        self.hyperParams = hyperParams

        self.action_space = action_space 

        self.buffer = deque()

        self.epsilon = self.hyperParams.START_EPSILON
        self.epsilon_decay_value = self.hyperParams.FIRST_EPSILON_DECAY
        
        self.gamma = self.hyperParams.GAMMA
        self.test = False

        self.device = torch.device("cuda" if cuda else "cpu")

        self.duelling = duelling
        self.cnn = cnn
        if(self.duelling):
            self.actor = DuellingActor(observation_space.shape[0], action_space.n, self.hyperParams, cnn=cnn).to(self.device) 
        else:
            self.actor = Actor(observation_space.shape[0], action_space.n, self.hyperParams, cnn=cnn).to(self.device) #for cartpole
        self.batch_size = self.hyperParams.BATCH_SIZE


        if(actor_to_load != None): #if it's a test, use the loaded NN
            self.epsilon = self.hyperParams.MIN_EPSILON
            self.actor.load_state_dict(torch.load(actor_to_load, map_location=self.device))
            self.actor.eval()
            self.test = True
        
        self.actor_target = copy.deepcopy(self.actor) #a target network is used to make the convergence possible (see papers on DRL)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR) # smooth gradient descent

        self.observation_space = observation_space

        self.double = double

        self.update_target = 0

        self.tab_max_q = []
        self.tab_loss = []

        self.loss = HuberLoss()



    def epsilon_decay(self):
        self.epsilon -= self.epsilon_decay_value
        if(self.epsilon_decay_value == self.hyperParams.FIRST_EPSILON_DECAY and self.epsilon<self.hyperParams.FIRST_MIN_EPSILON ):
            self.epsilon = self.hyperParams.FIRST_MIN_EPSILON
            self.epsilon_decay_value = self.hyperParams.SECOND_EPSILON_DECAY
        if(self.epsilon < self.hyperParams.MIN_EPSILON):
            self.epsilon = self.hyperParams.MIN_EPSILON

  
        


    def act(self, observation):
        if(not self.test and len(self.buffer) >= self.hyperParams.LEARNING_START):
            self.epsilon_decay()
        self.update_target += 1  

        observation = torch.tensor(np.expand_dims(observation, axis=0), device=self.device)
        tens_qvalue = self.actor(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            action = indices.item() #return it
        else:
            action = randint(0, tens_qvalue.size()[0]-1) #choose a random action

        return action, None


    def memorize(self, ob_prec, action, ob, reward, done, infos):
        self.buffer.append([ob_prec, action, ob, reward, not(done)]) 
        if(len(self.buffer)>self.hyperParams.BUFFER_SIZE):
            self.buffer.popleft()  
  

    def learn(self, n_iter=None):


        spl = sample(self.buffer, min(len(self.buffer), self.hyperParams.BATCH_SIZE))
        
        if(self.cnn):
            tens_state = torch.tensor(np.stack([i[0] for i in spl]))
            tens_state_next = torch.tensor(np.stack([i[2] for i in spl]))
        else:
            tens_state = torch.tensor([i[0] for i in spl])
            tens_state_next = torch.tensor([i[2] for i in spl])
        tens_action = torch.tensor([i[1] for i in spl]).squeeze().long()
        tens_reward = torch.tensor([i[3] for i in spl]).squeeze().float()
        tens_done = torch.tensor([i[4] for i in spl]).squeeze().bool()

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