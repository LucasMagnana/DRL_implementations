import torch
from torch import nn
from torch.autograd import Variable
from math import *
import numpy as np




class Actor(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, max_action=1, tanh=False): #for saved hyperparameters
        super(Actor, self).__init__()
        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.int = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)
        self.out = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)
        self.max_action = max_action
        self.tanh = tanh

    def forward(self, ob):
        ob = ob.float()
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(out))
        if(self.tanh):
            return torch.tanh(self.out(out)*self.max_action)
        else:
            return self.out(out)*self.max_action


def layer_init(layer, ppo, std=np.sqrt(2), bias_const=0.0):
    if(ppo):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    else:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    return layer


class ActorCritic(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, cnn=False, ppo=False, max_action=-1): #for saved hyperparameters
        super(ActorCritic, self).__init__()

        self.cnn = cnn
        self.ppo = ppo
        self.max_action = max_action

        if(cnn):
            self.network = CNN_layers(size_ob, hyperParams)
        else:
            l1 = layer_init(nn.Linear(np.array(size_ob).prod(), 64), ppo)

            l2 = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2), ppo)

            self.network = nn.Sequential(
                l1,
                nn.Tanh(),
                l2,
                nn.Tanh()
            )

        self.actor = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action), ppo, std=0.01, bias_const=1.0)

        self.critic = layer_init(nn.Linear(hyperParams.HIDDEN_SIZE_2, 1), ppo, std=1)

        if(max_action>0):
            self.stds = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)
            torch.nn.init.kaiming_normal_(self.stds.weight, nonlinearity="relu")



    def forward(self, ob):
        features = self.network(ob)

        if(self.cnn):
            values = self.critic(features[0])
            advantages = self.actor(features[1])

        else:
            advantages = self.actor(features)
            values = self.critic(features)

        if(self.ppo):
            if(self.max_action>0):
                stds = self.stds(features)
                return nn.functional.tanh(advantages)*self.max_action, nn.functional.sigmoid(stds), values
            else:
                return advantages, values
        else:
            return values + (advantages - advantages.mean())



class CNN_layers(nn.Module):

    def __init__(self, size_ob, hyperParams): #for saved hyperparameters
        super(CNN_layers, self).__init__()

        c1 = nn.Conv2d(size_ob, 32, 8, stride=4)
        torch.nn.init.kaiming_normal_(c1.weight, nonlinearity="relu")

        c2 = nn.Conv2d(32, 64, 4, stride=2)
        torch.nn.init.kaiming_normal_(c2.weight, nonlinearity="relu")

        c3 = nn.Conv2d(64, 64, 3, stride=1)
        torch.nn.init.kaiming_normal_(c3.weight, nonlinearity="relu")

        l1 = nn.Linear(3136, hyperParams.HIDDEN_SIZE_2*2)
        torch.nn.init.kaiming_normal_(l1.weight, nonlinearity="relu")

        self.hidden_size = hyperParams.HIDDEN_SIZE_2

        self.cnn = nn.Sequential(
            c1,
            nn.ReLU(),
            c2,
            nn.ReLU(),
            c3,
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            l1,
            nn.ReLU())


    def forward(self, ob):
        features = self.cnn(ob.float()/255)
        return torch.split(features, self.hidden_size, dim=1)




class Critic(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams):
        super(Critic, self).__init__()
        self.inp = nn.Linear(size_ob+size_action, hyperParams.HIDDEN_SIZE_1)
        self.int = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)
        self.out = nn.Linear(hyperParams.HIDDEN_SIZE_2, 1)

    def forward(self, ob, action):
        out = nn.functional.relu(self.inp(torch.cat((ob, action), dim=1)))
        out = nn.functional.relu(self.int(out))
        return self.out(out)


        

class REINFORCE_Model(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams):
        super(REINFORCE_Model, self).__init__()       
        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE)
        self.out = nn.Linear(hyperParams.HIDDEN_SIZE, size_action)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, ob):
        ob = ob.float()
        out = nn.functional.relu(self.inp(ob))
        out = self.sm(self.out(out))
        return out

class PPO_Actor(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams, max_action=-1):
        super(PPO_Actor, self).__init__()

        self.max_action = max_action
        if(max_action < 0):
            self.actor = nn.Sequential(
                nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1),
                nn.ReLU(),
                nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2),
                nn.ReLU(),
                nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action),
                nn.Softmax(dim=-1))
        else:
            self.actor = nn.Sequential(
                nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1),
                nn.ReLU(),
                nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2),
                nn.ReLU())
            self.expectation = nn.Sequential(
                nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action),
                nn.Tanh())
            self.std = nn.Sequential(
                nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action),
                nn.Sigmoid())


    
    def forward(self, ob):
        if(self.max_action < 0):
            return self.actor(ob.float())
        else:
            features = self.actor(ob.float())
            return self.expectation(features)*self.max_action, self.std(features)



class PPO_Critic(nn.Module):
    def __init__(self, size_ob, hyperParams, cnn=False):
        super(PPO_Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1),
                nn.ReLU(),
                nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2),
                nn.ReLU(),
                nn.Linear(hyperParams.HIDDEN_SIZE_2, 1)
                )
    
    def forward(self, ob):
        return self.critic(ob.float())