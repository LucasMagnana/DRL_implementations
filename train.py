import argparse
import sys
import gym 
import numpy as np
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
        env = gym.make(args.module, frameskip=4)
        env = ResizeObservation(env, shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=4)

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
        ob = env.reset()[0]
        prec_lives = 5
        sum_rewards=0
        steps=0
        nb_noop = 0
        while True:
            ob_prec = ob   
            action, infos = agent.act(ob)
            #print("\rAction: {}, Step: {}, sr: {}, noop: {}".format(action, steps, sum_rewards, nb_noop), end="")
            ob, reward, done, _, info = env.step(action)
            done_lives = True
            if("ALE" in args.module):
                reward = np.clip(reward, -1, 1)
                done_lives = prec_lives != info["lives"]
                prec_lives = info["lives"]
                if(action == 0):
                    nb_noop += 1
                else:
                    nb_noop = 0

            agent.memorize(ob_prec, action, ob, reward, done or done_lives, infos)
            sum_rewards += reward
            if(args.algorithm != "PPO" and steps%hyperParams.LEARN_EVERY == 0 and len(agent.buffer) > hyperParams.LEARNING_START):
                agent.learn()
            steps+=1
            if done or steps > hyperParams.MAX_STEPS or nb_noop > 30:
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
          

    save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams)

    # Close the env (only useful for the gym envs for now)
    env.close()