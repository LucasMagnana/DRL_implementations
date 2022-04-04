import torch
from torch import nn
from torch.autograd import Variable




class Actor(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, max_action=1, tanh=False): #for saved hyperparameters
        super(Actor, self).__init__()
        print(hyperParams)
        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE)
        self.int = nn.Linear(hyperParams.HIDDEN_SIZE, hyperParams.ACT_INTER)
        self.out = nn.Linear(hyperParams.ACT_INTER, size_action)
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




class PPO_Model(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams):
        super(PPO_Model, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(size_ob, hyperParams.HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(hyperParams.HIDDEN_SIZE, hyperParams.HIDDEN_SIZE), nn.ReLU())

        self.actor = nn.Sequential(
            nn.Linear(hyperParams.HIDDEN_SIZE, size_action), nn.Softmax(dim=-1))

        self.critic = nn.Linear(hyperParams.HIDDEN_SIZE, 1)
    
    def forward(self, ob):
        ob = ob.float()
        out = self.shared(ob)
        return self.actor(out), self.critic(out)






