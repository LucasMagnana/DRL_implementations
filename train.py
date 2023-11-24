import argparse
import sys
import gym 
import numpy as np
import random
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation

import datetime as dt

from python.TD3Agent import *
from python.DQNAgent import *
from python.PPOAgent import *
from python.hyperParams import *
from python.utils import save





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("-a", "--algorithm", type=str, default="3DQN")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("-m", "--module", type=str, default="LunarLanderContinuous-v2")
    #"LunarLanderContinuous-v2" #"Acrobot-v1" #"CartPole-v1" #"BipedalWalker-v3" 

    args = parser.parse_args()

    cuda=False
    args.cuda = args.cuda and torch.cuda.is_available()
    print("cuda:", args.cuda)
    if(args.cuda):
        print(torch.cuda.get_device_name(0))

    if(args.render):
        env = gym.make(args.module, render_mode="human") #gym env
    elif("ALE" in args.module):
        cnn = True
        env = gym.make(args.module, frameskip=1, obs_type="grayscale")
        env = ResizeObservation(env, shape=84)

    else:
        env = gym.make(args.module) #gym env
        cnn = False

    if(args.algorithm == "DQN"):
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, cnn=cnn)
    elif(args.algorithm == "3DQN"):
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, double=True, duelling=True, cnn=cnn)
    elif(args.algorithm == "TD3"):
        hyperParams = TD3HyperParams()
        agent = TD3Agent(env.observation_space, env.action_space, hyperParams, cuda=args.cuda)
    elif(args.algorithm == "PPO"):
        hyperParams = PPOHyperParams()
        agent = PPOAgent(env.observation_space, env.action_space, hyperParams, continuous_action_space=isinstance(env.action_space, gym.spaces.box.Box), cnn=cnn)

    print(env.action_space)
    tab_sum_rewards = []
    tab_mean_rewards = []
    total_steps = 0
    ep = 0
    while(total_steps < hyperParams.TRAINING_FRAMES):
        if(not args.no_save and ep > 0 and ep%100 == 0):
            save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams)
        if(args.algorithm == "PPO"):
            agent.start_episode()
        if("ALE" in args.module):
            env.reset()
            ob = []
            for _ in range(4):
                ob_env, reward_env, done, _, info_env = env.step(0)
                ob.append(ob_env.squeeze())
            ob = np.stack(ob)
            lives = info_env["lives"]
        else:
            ob = env.reset()[0]
        sum_rewards=0
        steps=0
        while True:
            ob_prec = ob   
            action, infos = agent.act(ob)
            #print("\rAction: {}, Step: {}, sr: {}, noop: {}".format(action, steps, sum_rewards, nb_noop), end="")
            if("ALE" in args.module):
                prec_lives = lives
                ob = []
                ob_inter = []
                reward = 0
                done = False
                for _ in range(hyperParams.REPEAT_ACTION*hyperParams.FRAME_SKIP):
                    if(not done):
                        ob_env, reward_env, done, _, info_env = env.step(action)
                        ob_inter.append(ob_env.squeeze())
                        reward += reward_env
                        lives = info_env["lives"]
                    else:
                        ob_inter.append(ob_inter[-1]) 

                for i in range(hyperParams.FRAME_SKIP, hyperParams.REPEAT_ACTION*hyperParams.FRAME_SKIP+hyperParams.FRAME_SKIP, hyperParams.FRAME_SKIP):
                    ob.append(np.maximum(ob_inter[i-1], ob_inter[i-2]))

                ob = np.stack(ob)
                done_lives = lives != prec_lives
                reward = np.clip(reward, -1, 1)
                agent.memorize(ob_prec, action, ob, reward, done or done_lives, infos)
            else:
                ob, reward, done, _, info = env.step(action)
                agent.memorize(ob_prec, action, ob, reward, done, infos)
            sum_rewards += reward
            if(args.algorithm != "PPO" and len(agent.buffer) > hyperParams.LEARNING_START):
                agent.learn()
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                if("DQN" not in args.algorithm):
                    agent.end_episode()
                    if(args.algorithm == "PPO" and ep > 0 and ep%hyperParams.LEARN_EVERY==0): #len(agent.batch_rewards) > hyperParams.MAXLEN):
                        agent.learn()

                tab_sum_rewards.append(sum_rewards)   
                tab_mean_rewards.append(np.mean(tab_sum_rewards[-100:]))
                total_steps += steps  
                ep += 1 
                break
        print("\rStep: {}, ep: {}, Average of last 100: {:.2f}".format(total_steps, ep, tab_mean_rewards[-1]), end="")
          

    save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams)

    # Close the env (only useful for the gym envs for now)
    env.close()