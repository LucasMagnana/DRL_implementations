import argparse
import sys
import gym 
import pickle
import numpy as np
from random import randint

from gym.wrappers import RecordVideo, FrameStack, ResizeObservation
import matplotlib.pyplot as plt

import datetime as dt

from python.utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("-a", "--algorithm", type=str, default="PPO")
    parser.add_argument("-m", "--module", type=str, default="LunarLanderContinuous-v2")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    cuda=False
    args.cuda = args.cuda and torch.cuda.is_available()
    print("cuda:", args.cuda)
    if(args.cuda):
        print(torch.cuda.get_device_name(0))


    if("ALE" in args.module):
        env = gym.make(args.module, frameskip=4, obs_type="grayscale", repeat_action_probability=0, render_mode="human", difficulty=1)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)

    elif(args.save):
        env = gym.make(args.module, render_mode="rgb_array") #gym en
        env = RecordVideo(env, video_folder='videos')
    else:
        env = gym.make(args.module, render_mode="human") #gym env

    actor_to_load = "./files/"+args.module.removeprefix("ALE/")+"_"+args.algorithm+".n"

    #load the hyper parameters
    with open("./files/"+args.module.removeprefix("ALE/")+"_"+args.algorithm+".hp", 'rb') as infile:
        hyperParams = pickle.load(infile)

    print(hyperParams.__dict__)

    hyperParams, agent = load_agent_and_hp(args, env, isinstance(env.action_space, gym.spaces.box.Box), actor_to_load=actor_to_load)

    
    tab_sum_rewards = []

    for e in range(1):
        ob = env.reset()[0]
        if("ALE" in args.module):
            for _ in range(randint(1, 30)):
                ob, reward, done, _, info = env.step(randint(1,2))
            ob = np.array(ob).squeeze()
        sum_rewards=0
        steps=0
        while True:
            ob_prec = ob   
            action, infos = agent.act(ob)
            ob, reward, done, _, _ = env.step(action)
            if("ALE" in args.module):
                ob = np.array(ob).squeeze()
            sum_rewards += reward
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                tab_sum_rewards.append(sum_rewards)            
                break

    env.close()


    print("Sum reward : ", sum_rewards) 