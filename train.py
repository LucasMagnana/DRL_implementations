import argparse
import sys
import gym 
import pickle
import numpy as np
from gym.wrappers import FrameStack, ResizeObservation
from random import randint

import matplotlib.pyplot as plt

import datetime as dt

from python.utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--algorithm", type=str, default="3DQN")
    parser.add_argument("-m", "--module", type=str, default="CartPole-v1")
    #"LunarLanderContinuous-v2" #"Acrobot-v1" #"CartPole-v1" #"BipedalWalker-v3" 

    args = parser.parse_args()

    cuda=False
    args.cuda = False #args.cuda and torch.cuda.is_available()
    print("cuda:", args.cuda)
    if(args.cuda):
        print(torch.cuda.get_device_name(0))
    
    
    if("ALE" in args.module):
        env = gym.make(args.module, frameskip=4, obs_type="grayscale", repeat_action_probability=0)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)

    else:
        env = gym.make(args.module) #gym env

    hyperParams, agent = load_agent_and_hp(args, env, isinstance(env.action_space, gym.spaces.box.Box))

    print(hyperParams.__dict__) 
    tab_sum_rewards = []
    tab_mean_rewards = []
    total_steps = 0
    ep = 0
    max_mean_reward = None
    while(total_steps < hyperParams.TRAINING_FRAMES):
        if(ep > 0 and ep%100 == 0):
            if(max_mean_reward == None or max_mean_reward < tab_mean_rewards[-1]):
                max_mean_reward = tab_mean_rewards[-1]
            save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams, max_mean_reward==tab_mean_rewards[-1])
        if(args.algorithm == "PPO"):
            agent.start_episode()
        ob = env.reset()[0]
        if("ALE" in args.module):
            for _ in range(randint(1, hyperParams.NOOP)):
                ob, reward, done, _, info = env.step(0)
        prec_lives = 5
        sum_rewards=0
        steps=0
        while True:
            ob_prec = ob   
            action, infos = agent.act(np.array(ob).squeeze())
            #print("\rAction: {}, Step: {}, sr: {}, noop: {}".format(action, steps, sum_rewards, nb_noop), end="")
            ob, reward, done, _, info = env.step(action)
            done_lives = False
            if("ALE" in args.module):
                done_lives = prec_lives != info["lives"]
                prec_lives = info["lives"]

            agent.memorize(np.array(ob_prec).squeeze(), action, np.array(ob).squeeze(), np.clip(reward, -1, 1), done or done_lives, infos)
            sum_rewards += reward
            if(args.algorithm != "PPO" and steps%hyperParams.LEARN_EVERY == 0 and len(agent.buffer) > hyperParams.LEARNING_START):
                agent.learn()
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                if("DQN" not in args.algorithm):
                    agent.end_episode()
                    if(args.algorithm == "PPO" and len(agent.batch_rewards) > hyperParams.MAXLEN):
                        agent.learn()

                tab_sum_rewards.append(sum_rewards)   
                tab_mean_rewards.append(np.mean(tab_sum_rewards[-100:]))
                total_steps += steps  
                ep += 1 
                break
        print("\rStep: {}, ep: {}, Average of last 100: {:.2f}".format(total_steps, ep, tab_mean_rewards[-1]), end="")
        
        agent.tab_max_q = []

    save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams, max_mean_reward<tab_mean_rewards[-1])

    # Close the env (only useful for the gym envs for now)
    env.close()