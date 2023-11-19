import argparse
import sys
import gym 
import pickle

import matplotlib.pyplot as plt

import datetime as dt

from python.TD3Agent import *
from python.DQNAgent import *
from python.PPOAgent import *
from python.hyperParams import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("-a", "--algorithm", type=str, default="PPO")
    parser.add_argument("-m", "--module", type=str, default="LunarLanderContinuous-v2")

    args = parser.parse_args()

    cuda=False
    args.cuda = args.cuda and torch.cuda.is_available()
    print("cuda:", args.cuda)
    if(args.cuda):
        print(torch.cuda.get_device_name(0))

    env = gym.make(args.module, render_mode="human") #gym env

    actor_to_load = "./trained_networks/"+args.module+"_"+args.algorithm+".n"

    #load the hyper parameters
    with open("./trained_networks/"+args.module+"_"+args.algorithm+".hp", 'rb') as infile:
        hyperParams = pickle.load(infile)

    if(args.algorithm == "DQN"):
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, actor_to_load=actor_to_load)
    elif(args.algorithm == "3DQN"):
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, double=True, duelling=True, actor_to_load=actor_to_load)
    elif(args.algorithm == "TD3"):
        agent = TD3Agent(env.observation_space, env.action_space, hyperParams, cuda=args.cuda, actor_to_load=actor_to_load)
    elif(args.algorithm == "PPO"):
        agent = PPOAgent(env.observation_space, env.action_space, hyperParams, continuous_action_space=isinstance(env.action_space, gym.spaces.box.Box),\
                         actor_to_load=actor_to_load)
    
    tab_sum_rewards = []

    for e in range(1):
        ob = env.reset()[0]
        sum_rewards=0
        steps=0
        while True:
            ob_prec = ob   
            action, infos = agent.act(ob)
            ob, reward, done, _, _ = env.step(action)
            sum_rewards += reward
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                tab_sum_rewards.append(sum_rewards)            
                break

    env.close()


    print("Sum reward : ", sum_rewards) 