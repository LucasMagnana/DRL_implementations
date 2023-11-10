import torch
from torch import nn
from torch.autograd import Variable




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



class DuellingActor(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, max_action=1, tanh=False): #for saved hyperparameters
        super(DuellingActor, self).__init__()

        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.feature_out = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)

        self.advantage_out = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)

        self.value_out = nn.Linear(hyperParams.HIDDEN_SIZE_2, 1)

        self.max_action = max_action
        self.tanh = tanh

    def forward(self, ob):
        ob = ob.float()
        features = nn.functional.relu(self.inp(ob))
        features = nn.functional.relu(self.feature_out(features))

        values = self.value_out(features)

        advantages = self.advantage_out(features)

        return values + (advantages - advantages.mean())




class Critic(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams):
        super(Critic, self).__init__()
        self.inp = nn.Linear(size_ob+size_action, hyperParams.CRIT_IN)
        self.int = nn.Linear(hyperParams.CRIT_IN, hyperParams.CRIT_INTER)
        self.out = nn.Linear(hyperParams.CRIT_INTER, 1)

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
                nn.Linear(size_ob, hyperParams.HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(hyperParams.HIDDEN_SIZE, size_action),
                nn.Softmax(dim=-1))
        else:
            self.actor = nn.Sequential(
                nn.Linear(size_ob, hyperParams.HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(hyperParams.HIDDEN_SIZE, size_action),
                nn.Tanh())

    
    def forward(self, ob):
        if(self.max_action < 0):
            return self.actor(ob.float())
        else:
            return self.actor(ob.float())*self.max_action



class PPO_Critic(nn.Module):
    def __init__(self, size_ob, hyperParams):
        super(PPO_Critic, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(size_ob, hyperParams.HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(hyperParams.HIDDEN_SIZE, 1)
                )
    
    def forward(self, ob):
        return self.critic(ob.float())



class PPO_Actor_CNN(nn.Module):
    def __init__(self, size_ob, size_action, max_action=-1):
        super(PPO_Critic_CNN, self).__init__()

        self.conv_1 = nn.Conv2d(size_ob[0], 16, 2)
        self.conv_2 = nn.Conv2d(16, 16, 2)

        out_shape = 1728

        self.max_action = max_action

        if(max_action < 0):
            self.actor = nn.Sequential(
                nn.Tanh(),
                nn.Linear(out_shape, 128),
                nn.Tanh(),
                nn.Linear(128, size_action),
                nn.Softmax(dim=-1))
        else:
            self.actor = nn.Sequential(
                nn.Tanh(),
                nn.Linear(out_shape_actor, 64),
                nn.Tanh(),
                nn.Linear(64, size_action),
                nn.Tanh())

    
    def forward(self, ob):
        ob = ob.float()
        features = self.conv_1(ob)
        features = self.conv_2(features)
        if(len(features.shape) == 3):
            features = torch.flatten(features)
        elif(len(features.shape) == 4):
            features = torch.flatten(features, start_dim=1)
        #features = nn.functional.relu(self.out(features))
        return self.actor(features)



class PPO_Critic_CNN(nn.Module):
    def __init__(self, size_ob):
        super(PPO_Critic_CNN, self).__init__()

        self.conv_1 = nn.Conv2d(size_ob[0], 16, 2)
        self.conv_2 = nn.Conv2d(16, 16, 2)

        out_shape = 1728

        self.critic = nn.Sequential(
                nn.Tanh(),
                nn.Linear(out_shape, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
                )
    
    def forward(self, ob):
        ob = ob.float()
        features = self.conv_1(ob)
        features = self.conv_2(features)
        if(len(features.shape) == 3):
            features = torch.flatten(features)
        elif(len(features.shape) == 4):
            features = torch.flatten(features, start_dim=1)
        #features = nn.functional.relu(self.out(features))
        return self.critic(features)











