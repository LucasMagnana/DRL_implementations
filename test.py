import argparse
import sys
import gym 
import pickle
import numpy as np
from random import randint

from gym.wrappers import RecordVideo, FrameStack, ResizeObservation
import stable_baselines3.common.atari_wrappers as atari_wrappers


import matplotlib.pyplot as plt

import datetime as dt

from python.utils import *



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("-a", "--algorithm", type=str, default="PPO")
    parser.add_argument("-m", "--module", type=str, default="CartPole-v1")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    cuda=False
    args.cuda = args.cuda and torch.cuda.is_available()
    print("cuda:", args.cuda)
    if(args.cuda):
        print(torch.cuda.get_device_name(0))

    render_mode = "human"
    if(args.save):
        render_mode="rgb_array"


    if("ALE" in args.module):
        if(args.algorithm == "3DQN"):
            env = gym.make(args.module, frameskip=4, obs_type="grayscale", repeat_action_probability=0, render_mode=render_mode)
            env = ResizeObservation(env, shape=84)
            env = FrameStack(env, num_stack=4)
        else:
            env = gym.make(args.module, frameskip=1, repeat_action_probability=0, render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = atari_wrappers.NoopResetEnv(env, noop_max=30)
            env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = atari_wrappers.FireResetEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
    else:
        env = gym.make(args.module, render_mode=render_mode) #gym env

    if(args.save):
        env = RecordVideo(env, video_folder=render_mode)

    actor_to_load = "./files/"+args.module.removeprefix("ALE/")+"_"+args.algorithm+".n"

    #load the hyper parameters
    with open("./files/"+args.module.removeprefix("ALE/")+"_"+args.algorithm+".hp", 'rb') as infile:
        hyperParams = pickle.load(infile)

    print(hyperParams.__dict__)

    agent = load_agent(args, env, isinstance(env.action_space, gym.spaces.box.Box), hyperParams, actor_to_load=actor_to_load)

    
    tab_sum_rewards = []

    for e in range(1):
        ob = env.reset()[0]
        sum_rewards=0
        steps=0
        done = False
        while not done:
            if("ALE/" in args.module and args.algorithm == "3DQN"):
                ob = torch.tensor(ob).squeeze(-1)
            ob = np.expand_dims(ob, 0)
            action, infos = agent.act(ob)
            ob, reward, done, _, infos = env.step(action.squeeze().numpy())
            sum_rewards += reward
            steps+=1


    env.close()


    print("Sum reward : ", sum_rewards) 