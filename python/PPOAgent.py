import numpy as np

import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from python.NeuralNetworks import PPO_Actor, PPO_Critic

'''
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)
'''

def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def discount_rewards(rewards, list_done, gamma):
    r = []
    for reward, done in zip(reversed(rewards), reversed(list_done)):
        if done:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        r.insert(0, discounted_reward)
    r = np.array(r, dtype=np.single)
    return r - r.mean()


def gae(rewards, values, episode_ends, gamma, lam):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages




class PPOAgent():

    def __init__(self, ob_space, ac_space, hyperParams, model_to_load=None, continuous_action_space=False):

        self.hyperParams = hyperParams

        self.continuous_action_space = continuous_action_space


        if(self.continuous_action_space):
            self.ac_space = ac_space.shape[0]
            self.actor = PPO_Actor(ob_space, self.ac_space, hyperParams, max_action=ac_space.high[0])
        else:
            self.ac_space = ac_space.n  
            self.actor = PPO_Actor(ob_space, self.ac_space, hyperParams)

        if(model_to_load != None):
            self.actor.load_state_dict(torch.load(model_to_load))
            self.actor.eval()


        self.critic = PPO_Critic(ob_space, hyperParams)

        # Define optimizer
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.hyperParams.LR)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.hyperParams.LR)
        self.mse = torch.nn.MSELoss()

        

        if(self.continuous_action_space):
            self.action_std = torch.full((self.ac_space,), 1/100)

        self.num_decisions_made = 0

        self.tab_losses = []

        self.reset_batches()


    def reset_batches(self):
        self.batch_rewards = []
        self.batch_advantages = []
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_selected_probs = []
        self.batch_done = []


    def learn(self):

        for k in range(self.hyperParams.K):

            state_tensor = torch.tensor(self.batch_states)

            #print(state_tensor.tolist() == self.batch_states)

            advantages_tensor = torch.tensor(self.batch_advantages)
            old_selected_probs_tensor = torch.tensor(self.batch_selected_probs)

            old_values_tensor = torch.tensor(self.batch_values)

            rewards_tensor = torch.tensor(self.batch_rewards)
            # Normalizing the rewards:
            #rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-5)
            

            action_tensor = torch.tensor(self.batch_actions)

            
            # Calculate actor loss
            probs = self.actor(state_tensor)
            values_tensor = self.critic(state_tensor)

            values_tensor = values_tensor.flatten()

            if(self.continuous_action_space):
                print(probs)
                action_std = self.action_std.expand_as(probs)
                mat_std = torch.diag_embed(action_std)
                dist = MultivariateNormal(probs, mat_std)
                selected_probs_tensor = dist.log_prob(action_tensor.flatten())
                ratios = torch.exp(selected_probs_tensor - old_selected_probs_tensor.detach())

            else:
                # Actions are used as indices, must be 
                # LongTensor
                action_tensor = action_tensor.long()
                selected_probs_tensor = torch.index_select(probs, 1, action_tensor).diag()
                ratios = selected_probs_tensor/old_selected_probs_tensor

            
            #advantages_tensor = rewards_tensor - values_tensor.detach()   

            loss_actor = ratios*advantages_tensor
            clipped_loss_actor = torch.clamp(ratios, 1-self.hyperParams.EPSILON, 1+self.hyperParams.EPSILON)*advantages_tensor

            loss_actor = -(torch.min(loss_actor, clipped_loss_actor).mean())

            # Calculate critic loss
            '''value_pred_clipped = old_values_tensor + (values_tensor - old_values_tensor).clamp(-self.hyperParams.EPSILON, self.hyperParams.EPSILON)
            value_losses = (values_tensor - rewards_tensor) ** 2
            value_losses_clipped = (value_pred_clipped - rewards_tensor) ** 2

            loss_critic = 0.5 * torch.max(value_losses, value_losses_clipped)
            loss_critic = loss_critic.mean()''' 


            loss_critic = self.mse(values_tensor, rewards_tensor)
         


            self.tab_losses.append((loss_actor.item()+loss_critic.item())/2)

            #print("Loss :", self.tab_losses[-1])

            # Reset gradients
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            # Calculate gradients
            loss_actor.backward()
            loss_critic.backward()
            # Apply gradients
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.reset_batches()


    def memorize(self, ob_prec, val, action_probs, action, ob, reward, done):
        self.states.append(ob_prec)
        self.values.extend(val)
        self.rewards.append(reward)
        self.actions.append(action)
        if(self.continuous_action_space):
            self.selected_probs.append(action_probs.item())
        else:
            self.selected_probs.append(action_probs[action])
        self.list_done.append(done)  

    def start_episode(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.values = []
        self.selected_probs = []
        self.list_done = []

    def end_episode(self):
        self.list_done[-1] = True
        self.batch_rewards.extend(discount_rewards(self.rewards, self.list_done, self.hyperParams.GAMMA))
        gaes = gae(np.expand_dims(np.array(self.rewards), 0), np.expand_dims(np.array(self.values), 0), np.expand_dims(np.array([not elem for elem in self.list_done]), 0), self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
        self.batch_advantages.extend(gaes[0])
        self.batch_states.extend(self.states)
        self.batch_values.extend(self.values)
        self.batch_actions.extend(self.actions)
        self.batch_done.extend(self.list_done)
        self.batch_selected_probs.extend(self.selected_probs)


    def act(self, observation):
        # Get actions and convert to numpy array
        action_probs = self.actor(torch.tensor(observation))
        val = self.critic(torch.tensor(observation))
        val = val.detach().numpy()
        if(self.continuous_action_space):
            std_mat = torch.diag(self.action_std)
            dist = MultivariateNormal(action_probs, std_mat)
            action = dist.sample()
            action_probs = dist.log_prob(action).detach().numpy()
            action = action.detach().numpy()
        else:
            action_probs = action_probs.detach().numpy()
            action = np.random.choice(np.arange(self.ac_space), p=action_probs)

        self.num_decisions_made += 1

        return action, val, action_probs