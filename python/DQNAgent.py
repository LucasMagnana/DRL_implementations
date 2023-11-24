
import random
from random import sample, random, randint

from torch.nn import MSELoss, HuberLoss
import torchrl

import numpy as np

import copy
import numpy as np
import torch

from python.NeuralNetworks import Actor, DuellingActor, DuellingActor_CNN, Actor_CNN
from python.utils import *

class DQNAgent(object):
    def __init__(self, observation_space, action_space, hyperParams, test=False, double=False, duelling=False, PER=False, cuda=False,\
    actor_to_load=None, cnn=False):

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

        #self.tau = self.hyperParams.TAU
        self.epsilon = self.hyperParams.EPSILON
        self.gamma = self.hyperParams.GAMMA

        self.device = torch.device("cuda" if cuda else "cpu")

        self.duelling = duelling
        self.cnn = cnn
        if(self.cnn):
            self.list_modifs = []
            self.original_image = None
            if(self.duelling):
                self.actor = DuellingActor_CNN(observation_space.shape[0], action_space.n, self.hyperParams).to(self.device) 
            else:
                self.actor = Actor_CNN(observation_space.shape[0], action_space.n, self.hyperParams).to(self.device) 
        elif(self.duelling):
            self.actor = DuellingActor(observation_space.shape[0], action_space.n, self.hyperParams).to(self.device) 
        else:
            self.actor = Actor(observation_space.shape[0], action_space.n, self.hyperParams).to(self.device) #for cartpole

        self.batch_size = self.hyperParams.BATCH_SIZE


        if(actor_to_load != None): #if it's a test, use the loaded NN
            self.actor.load_state_dict(torch.load(actor_to_load, map_location=self.device))
            self.actor.eval()
        
        self.actor_target = copy.deepcopy(self.actor) #a target network is used to make the convergence possible (see papers on DRL)

        self.optimizer = torch.optim.RMSprop(self.actor.parameters(), self.hyperParams.LR, alpha=0.95, momentum=0.95, eps=0.1) # smooth gradient descent

        self.observation_space = observation_space

        self.double = double

        self.target_update_counter = 0

        self.loss = HuberLoss()
        
        


    def act(self, observation):
        #return self.action_space.sample()
        observation = torch.tensor(np.array(observation), device=self.device)
        tens_qvalue = self.actor(observation) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            action = indices.item() #return it
        else:
            action = randint(0, tens_qvalue.size()[0]-1) #choose a random action
        self.target_update_counter += 1
        return action, None

    def sample(self):
        if(len(self.buffer) < self.batch_size):
            return self.buffer.sample(len(self.buffer))
        else:
            return self.buffer.sample(self.batch_size)
            

    def memorize(self, ob_prec, action, ob, reward, done, infos):
        if(self.cnn):
            experience = copy.deepcopy(ob_prec)
        else:
            experience = copy.deepcopy(ob_prec).flatten()
        experience = np.append(experience, action)
        if(self.cnn):
            experience = np.append(experience, ob)
        else:
            experience = np.append(experience, ob.flatten())
        experience = np.append(experience, reward)
        experience = np.append(experience, not(done))
        self.buffer.add(torch.ByteTensor(experience, device=self.device))   

    '''def memorize(self, ob_prec, action, ob, reward, done, infos):
        if(self.cnn):
            if(self.original_image == None):
                self.original_image = torch.tensor(np.array(ob))
            modifs_ob = compute_modifications(self.original_image, torch.tensor(np.array(ob)))
            modifs_ob_prec = compute_modifications(self.original_image, torch.tensor(np.array(ob_prec)))
            if(self.memorize_step < self.hyperParams.BUFFER_SIZE):
                self.buffer.append([torch.ByteTensor(np.array(ob_prec)), action, torch.ByteTensor(np.array(ob)), reward, not(done)])
            else:
                self.buffer[self.memorize_step%self.hyperParams.BUFFER_SIZE] = [torch.ByteTensor(np.array(ob_prec)), action,\
                torch.ByteTensor(np.array(ob)), reward, not(done)]
            self.memorize_step += 1'''


    def learn(self, n_iter=None):
        #actual noise decaying method, works well with the custom env
        self.epsilon -= self.hyperParams.EPSILON_DECAY
        if(self.epsilon<self.hyperParams.MIN_EPSILON):
            self.epsilon=self.hyperParams.MIN_EPSILON


        spl = self.sample()  #create a batch of experiences
        if(self.PER):
            datas = spl[1]
            spl = spl[0]

        if(self.cnn):
            spl = torch.split(spl, [4*84*84, 1, 4*84*84, 1, 1], dim=1)
            tens_state = torch.reshape(spl[0], [32, 4, 84, 84])
            tens_state_next = torch.reshape(spl[2], [32, 4, 84, 84])
        else:
            spl = torch.split(spl, [self.observation_space.shape[0], 1, self.observation_space.shape[0], 1, 1], dim=1)
            tens_state = spl[0]
            tens_state_next = spl[2]



        tens_action = spl[1].squeeze().long()

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
        tens_loss = self.loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient


        if(self.target_update_counter >= self.hyperParams.TARGET_UPDATE):
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.target_update_counter = 0

        '''for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()): #updates the target network
            target_param.data.copy_(self.tau * param + (1-self.tau)*target_param )'''


'''
CODE FOR TORCHRL EXPERIENCE REPLAY

    def memorize_ER(self, ob_prec, action, ob, reward, done, infos):
        if(self.cnn):
            if(self.original_image == None):
                self.original_image = torch.tensor(ob)
            modifs_ob = compute_modifications(self.original_image, torch.tensor(ob))
            modifs_ob_prec = compute_modifications(self.original_image, torch.tensor(ob_prec))
            self.list_modifs.append(modifs_ob_prec)
            self.list_modifs.append(modifs_ob)
            experience = torch.tensor(len(self.list_modifs)-2)
            experience = np.append(experience, action)
            experience = np.append(experience, len(self.list_modifs)-1)
        else:
            experience = copy.deepcopy(ob_prec).flatten()
            experience = np.append(experience, action)
            experience = np.append(experience, ob.flatten())
            
        experience = np.append(experience, reward)
        experience = np.append(experience, not(done))
        self.buffer.add(torch.FloatTensor(experience, device=self.device))   

SAMPLE :
spl = self.sample()  #create a batch of experiences
        if(self.PER):
            datas = spl[1]
            spl = spl[0]

        if(self.cnn):
            spl = torch.split(spl, [1, 1, 1, 1, 1], dim=1)
            for i in range(len(spl[0])):
                state = modify_image(self.original_image, self.list_modifs[int(spl[0][i].item())])
                state_next = modify_image(self.original_image, self.list_modifs[int(spl[2][i].item())])
                if(i == 0):
                    tens_state = state
                    tens_state_next = state_next
                else:
                    tens_state = torch.cat((tens_state, state))
                    tens_state_next = torch.cat((tens_state_next, state_next))
            tens_state = torch.reshape(tens_state, (32, 4, 84, 84))
            tens_state_next = torch.reshape(tens_state_next, (32, 4, 84, 84))
        else:
            spl = torch.split(spl, [self.observation_space.shape[0], 1, self.observation_space.shape[0], 1, 1], dim=1)
            tens_state = spl[0]
            tens_state_next = spl[2]



        tens_action = spl[1].squeeze().long()

        tens_reward = spl[3].squeeze()

        tens_done = spl[4].squeeze().bool()
'''
        