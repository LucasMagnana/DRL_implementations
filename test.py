import argparse
import sys
import gym 
import pickle

import matplotlib.pyplot as plt

import datetime as dt

from python.DQNAgent import *






def test(module="CartPole-v1"):

    cuda = torch.cuda.is_available()
    
    env = gym.make(module, render_mode="human")

    with open('./trained_networks/'+module+'.hp', 'rb') as infile:
        hyperParams = pickle.load(infile)
    
    agent = DQNAgent(env.action_space, env.observation_space, hyperParams, test=True, cuda=cuda, actor_to_load='./trained_networks/'+module+'.n')

    tab_sum_rewards = []

    for e in range(1):
        ob = env.reset()[0]
        sum_rewards=0
        steps=0
        while True:
            env.render()
            ob_prec = ob   
            action = agent.act(ob)
            ob, reward, done, _, _ = env.step(action)
            sum_rewards += reward
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                tab_sum_rewards.append(sum_rewards)            
                break

    env.close()


    print("Sum reward : ", sum_rewards) 


if __name__ == '__main__':
    test()