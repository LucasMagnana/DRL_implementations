import argparse
import sys
import gym 
import pickle
import copy
import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from python.NeuralNetworks import PPO_Model
from python.hyperParams import PPOHyperParams, module
from python.PPOAgent import PPOAgent, get_screen

import time




if __name__ == '__main__':
    list_envs = []

    testing=False
    cnn = False
    

    hyperParams = PPOHyperParams()

    model_to_load=None

    env = gym.make(module).unwrapped

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "--test"):
            testing = True
            with open('./trained_networks/'+module+'_PPO.hp', 'rb') as infile:
                hyperParams = pickle.load(infile)

            model_to_load='./trained_networks/'+module+'_PPO.n'

            hyperParams.EPISODE_COUNT=1
            hyperParams.NUM_AGENTS=1
            hyperParams.K=0
            hyperParams.NUM_EP_ENV=1

            env = gym.make(module, render_mode="human").unwrapped #gym env

    env.reset()
    env.reset()

    if(cnn):
        init_screen = get_screen(env)
        _, _, screen_height, screen_width = init_screen.shape
        ppo_agent = PPOAgent(hyperParams, (screen_width, screen_height), env.action_space.n, model_to_load, cnn)
    else:
        ppo_agent = PPOAgent(hyperParams, env.observation_space.shape[0], env.action_space.n, model_to_load)

    for ep in range(hyperParams.EPISODE_COUNT):
        # Set up lists to hold results
        ppo_agent.reset_batches()


        for a in range(hyperParams.NUM_AGENTS):
            if(len(list_envs) < hyperParams.NUM_AGENTS):
                list_envs.append(gym.make(module))

            ppo_agent.act(env, render=testing)

        if(not testing):
            ppo_agent.learn()
            
    if(not testing):
        print()
        
    env.close()

    if(testing):
        print("Sum of rewards:", ppo_agent.total_rewards[-1])
    else:
        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure(figsize=(25, 12), dpi=80)
        plt.plot(ppo_agent.total_rewards, linewidth=1)
        plt.plot(ppo_agent.avg_rewards, linewidth=1)
        plt.ylabel('Sum of the rewards')       
        plt.savefig("./images/"+module+"_PPO.png")
        
        #save the neural networks of the policy
        torch.save(ppo_agent.model.state_dict(), './trained_networks/'+module+'_PPO.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/'+module+'_PPO.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
                

                