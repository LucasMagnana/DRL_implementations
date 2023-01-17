
import random
from random import sample, random, randint

from torch.nn import MSELoss
import torchrl

import numpy as np

import copy
import numpy as np
import torch

from python.NeuralNetworks import Actor, DuellingActor

class DQNAgent(object):
    def __init__(self, action_space, observation_space, hyperParams, test=False, double=True, duelling=True, PER=False, cuda=False, actor_to_load=None):

        self.hyperParams = hyperParams
        
        if(actor_to_load != None): #use the good hyper parameters (loaded if it's a test, written in the code if it's a training)
            self.hyperParams.EPSILON = 0

        self.action_space = action_space 
        
        self.PER = PER
        if(not test):
            if(self.PER):  
                self.buffer = torchrl.data.PrioritizedReplayBuffer(int(self.hyperParams.BUFFER_SIZE), 0.2, 0.4)
            else:
                self.buffer = torchrl.data.ReplayBuffer(int(self.hyperParams.BUFFER_SIZE))

        self.alpha = self.hyperParams.ALPHA
        self.epsilon = self.hyperParams.EPSILON
        self.gamma = self.hyperParams.GAMMA

        self.device = torch.device("cuda" if cuda else "cpu")

        self.duelling = duelling
        if(self.duelling):
            self.actor = DuellingActor(observation_space.shape[0], action_space.n, self.hyperParams).to(self.device) #for cartpole
        else:
            self.actor = Actor(observation_space.shape[0], action_space.n, self.hyperParams).to(self.device) #for cartpole

        self.batch_size = self.hyperParams.BATCH_SIZE


        if(actor_to_load != None): #if it's a test, use the loaded NN
            self.actor.load_state_dict(torch.load(actor_to_load, map_location=self.device))
            self.actor.eval()
        
        self.actor_target = copy.deepcopy(self.actor) #a target network is used to make the convergence possible (see papers on DRL)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR) # smooth gradient descent

        self.observation_space = observation_space

        self.double = double
        
        


    def act(self, observation):
        #return self.action_space.sample()
        observation = torch.tensor(observation, device=self.device)
        tens_qvalue = self.actor(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            return indices.item() #return it
        return randint(0, tens_qvalue.size()[0]-1) #choose a random action

    def sample(self):
        if(len(self.buffer) < self.batch_size):
            return self.buffer.sample(len(self.buffer))
        else:
            return self.buffer.sample(self.batch_size)

    def memorize(self, ob_prec, action, ob, reward, done):
        experience = copy.deepcopy(ob_prec).flatten()
        experience = np.append(experience, action)
        experience = np.append(experience, ob.flatten())
        experience = np.append(experience, reward)
        experience = np.append(experience, not(done))
        self.buffer.add(torch.FloatTensor(experience, device=self.device))   

    def learn(self, n_iter=None):

        #previous noise decaying method, works well with cartpole
        '''if(self.epsilon > self.hyperParams.MIN_EPSILON):
            self.epsilon *= self.hyperParams.EPSILON_DECAY
        else:
            self.epsilon = 0'''

        
        #actual noise decaying method, works well with the custom env
        self.epsilon -= self.hyperParams.EPSILON_DECAY
        if(self.epsilon<self.hyperParams.MIN_EPSILON):
            self.epsilon=self.hyperParams.MIN_EPSILON

        loss = MSELoss()

        spl = self.sample()  #create a batch of experiences
        if(self.PER):
            datas = spl[1]
            spl = spl[0]


        spl = torch.split(spl, [self.observation_space.shape[0], 1, self.observation_space.shape[0], 1, 1], dim=1)


        tens_state = spl[0]

        tens_action = spl[1].squeeze().long()

        tens_state_next = spl[2]

        tens_reward = spl[3].squeeze()

        tens_done = spl[4].squeeze().bool()

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
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()): #updates the target network
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )