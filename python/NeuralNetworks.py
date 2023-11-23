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



class DuellingActor_CNN(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams): #for saved hyperparameters
        super(DuellingActor_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(size_ob, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU())

        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU())

        self.advantage_out = nn.Linear(512, size_action)

        self.value_out = nn.Linear(512, 1)

    def forward(self, ob):
        features = self.cnn(ob.float())
        if(len(ob.shape)==4):
            features = torch.flatten(features, start_dim=1)
        else:
            features = torch.flatten(features)

        features = self.linear(features)
        values = self.value_out(features)

        advantages = self.advantage_out(features)

        return values + (advantages - advantages.mean())




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
    


class PPO_Model_CNN(nn.Module):
    def __init__(self, size_ob, size_action):
        super(PPO_Model_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(size_ob, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU())
        
        self.features = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU())
        print(size_action, nn.Linear(512, size_action))
        self.actor = nn.Linear(512, size_action)
        self.critic = nn.Linear(512, 1)


            
    def forward(self, ob):
        features = self.cnn(ob.float())
        if(len(ob.shape)==4):
            features = torch.flatten(features, start_dim=1)
        else:
            features = torch.flatten(features)
        features = self.features(features)
        return nn.functional.softmax(self.actor(features)), self.critic(features)






class Actor_CNN(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams, ppo=False):
        super(Actor_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(size_ob, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU())
        
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, size_action))

        if(ppo):
            self.actor.append(nn.Softmax(dim=-1))


    
    def forward(self, ob):
        features = self.cnn(ob.float())
        if(len(ob.shape)==4):
            features = torch.flatten(features, start_dim=1)
        else:
            features = torch.flatten(features)
        return self.actor(features)


class Critic_CNN(nn.Module):
    def __init__(self, size_ob, hyperParams):
        super(Critic_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(size_ob, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU())

        self.critic=nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
    
    def forward(self, ob):
        features = self.cnn(ob.float())
        if(len(ob.shape)==4):
            features = torch.flatten(features, start_dim=1)
        else:
            features = torch.flatten(features)
        return self.critic(features)
