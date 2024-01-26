import argparse
import sys
import gym 
import pickle
import numpy as np
import random
import torch

import matplotlib.pyplot as plt

import stable_baselines3.common.atari_wrappers as atari_wrappers

import datetime as dt

from python.utils import *

def make_env(module, seed):
    def thunk():
        if("ALE" in module):
            env = gym.make(module, frameskip=1, repeat_action_probability=0, autoreset=True)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = atari_wrappers.NoopResetEnv(env, noop_max=30)
            env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
            env = atari_wrappers.EpisodicLifeEnv(env)
            env = atari_wrappers.ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        else:
            env = gym.make(module, autoreset=True)
            env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        return env
    return thunk


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

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cpu")

    hyperParams = load_hp(args)

    if("NUM_ENV" not in hyperParams.__dict__):
        hyperParams.NUM_ENV = 1

    if(hyperParams.NUM_ENV > 1):
        env = gym.vector.SyncVectorEnv(
        [make_env(args.module, seed+i) for i in range(hyperParams.NUM_ENV)])
    else:
        env = make_env(args.module, seed)()

    agent = load_agent(args, env, isinstance(env.action_space, gym.spaces.box.Box), hyperParams)

    print(hyperParams.__dict__) 
    tab_sum_rewards = []
    tab_mean_rewards = []
    total_steps = 0
    ep = 0
    max_mean_reward = None

    sum_rewards = [0]
    steps = [0]

    num_update = 0
    env_num = 0


    ob = env.reset(seed=seed)[0]
    last_done = np.zeros(hyperParams.NUM_ENV)

    while(total_steps < hyperParams.TRAINING_FRAMES):
        ob_prec = ob
        action, agent_infos = agent.act(ob)
        agent_infos.append(env_num)
        agent_infos.append((total_steps//hyperParams.NUM_ENV)%hyperParams.BATCH_SIZE)
        ob, reward, done, _, infos = env.step(action.numpy())
        if(args.algorithm == "PPO"):
            save_done = last_done
        else:
            save_done = done
        agent.memorize(ob_prec, action, ob, reward, save_done, agent_infos)
        last_done = done
        sum_rewards[env_num] += reward
        if(args.algorithm != "PPO" and steps[env_num]%hyperParams.LEARN_EVERY == 0 and agent.num_transition_stored%hyperParams.BUFFER_SIZE > hyperParams.LEARNING_START):
            agent.learn()
            num_update += 1
        steps[env_num]+=1
        if(len(infos) > 0):
            episode_end = False
            if("episode" in infos):
                episode_end = True
                tab_sum_rewards.append(infos["episode"]["r"])
            elif("final_info" in infos):
                for elem in infos["final_info"]:
                    print(elem)
                    if elem != None and "episode" in elem:
                        tab_sum_rewards.append(elem["episode"]["r"])
                        episode_end = True
            if(episode_end):
                ep+=1
                tab_mean_rewards.append(np.mean(tab_sum_rewards[-100:]))
                if(ep > 0 and ep%100 == 0):
                    if(max_mean_reward == None or max_mean_reward < tab_mean_rewards[-1]):
                        max_mean_reward = tab_mean_rewards[-1]
                    save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams, max_mean_reward==tab_mean_rewards[-1])
                print("\rStep: {}, update: {}, ep: {}, Average of last 100: {:.2f}".format(total_steps, num_update, ep, tab_mean_rewards[-1]), end="")

        total_steps += hyperParams.NUM_ENV  
        
        if(args.algorithm == "PPO" and total_steps/hyperParams.NUM_ENV%hyperParams.BATCH_SIZE == 0):
            agent.learn(last_done, ob)
            num_update += 1
            if(total_steps <= hyperParams.TRAINING_FRAMES):
                agent.optimizer.param_groups[0]["lr"] = (hyperParams.TRAINING_FRAMES-total_steps)/hyperParams.TRAINING_FRAMES*hyperParams.LR
                agent.lr.append(agent.optimizer.param_groups[0]["lr"])

    save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams, max_mean_reward<tab_mean_rewards[-1])

    # Close the env (only useful for the gym envs for now)
    env.close()