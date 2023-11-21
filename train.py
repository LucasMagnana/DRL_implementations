import argparse
import sys
import gym 
import pickle
import numpy as np
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation

import matplotlib.pyplot as plt

import datetime as dt

from python.TD3Agent import *
from python.DQNAgent import *
from python.PPOAgent import *
from python.hyperParams import *


def save(tab_sum_rewards, tab_mean_rewards, module, args, agent, hyperParams):
    #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
    plt.clf()
    plt.figure()
    plt.plot(tab_sum_rewards, alpha=0.75)
    plt.plot(tab_mean_rewards, color="darkblue")
    plt.xlabel('Episodes')   
    plt.ylabel('Sum of rewards')       
    plt.savefig("./images/"+module+"_"+args.algorithm+".png")
    
    #save the neural networks of the agent
    print()
    print("Saving...")
    #torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'_target.n')
    torch.save(agent.actor.state_dict(), "./trained_networks/"+module+"_"+args.algorithm+".n")

    #save the hyper parameters (for the tests and just in case)
    with open("./trained_networks/"+module+"_"+args.algorithm+".hp", 'wb') as outfile:
        pickle.dump(hyperParams, outfile)






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

    if(args.algorithm == "DQN"):
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, cnn=cnn)
    elif(args.algorithm == "3DQN"):
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, double=True, duelling=True)
    elif(args.algorithm == "TD3"):
        hyperParams = TD3HyperParams()
        agent = TD3Agent(env.observation_space, env.action_space, hyperParams, cuda=args.cuda)
    elif(args.algorithm == "PPO"):
        hyperParams = PPOHyperParams()
        agent = PPOAgent(env.observation_space, env.action_space, hyperParams, continuous_action_space=isinstance(env.action_space, gym.spaces.box.Box), cnn=cnn)


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
        sum_rewards=0
        steps=0
        nb_noop_at_start = 0
        while True:
            ob_prec = ob   
            action, infos = agent.act(ob)
            ob, reward, done, _, _ = env.step(action)
            if("ALE" in args.module):
                reward = np.clip(reward, -1, 1)
                if(nb_noop_at_start >= 0):
                    if(action == 0):
                        nb_noop_at_start += 1
                    else:
                        nb_noop_at_start = -1
                    if(nb_noop_at_start == 30):
                        reward = -10
            agent.memorize(ob_prec, action, ob, reward, done, infos)
            sum_rewards += reward
            if(args.algorithm != "PPO" and steps%hyperParams.LEARN_EVERY == 0 and len(agent.buffer) > hyperParams.LEARNING_START):
                agent.learn()
            steps+=1
            if done or steps > hyperParams.MAX_STEPS or nb_noop_at_start == 30:
                if("DQN" not in args.algorithm):
                    agent.end_episode()
                    if(args.algorithm == "PPO" and ep > 0 and ep%hyperParams.NUM_EP_ENV == 0):
                        agent.learn()

                tab_sum_rewards.append(sum_rewards)   
                tab_mean_rewards.append(np.mean(tab_sum_rewards[-100:]))
                total_steps += steps  
                ep += 1 
                break

        print("\rStep: {} Average of last 100: {:.2f}".format(total_steps, tab_mean_rewards[-1]), end="")
          

    save(tab_sum_rewards, tab_mean_rewards, args.module.removeprefix("ALE/"), args, agent, hyperParams)

    # Close the env (only useful for the gym envs for now)
    env.close()