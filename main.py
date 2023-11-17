import argparse
import sys
import gym 
import pickle
import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from python.TD3Agent import *
from python.DQNAgent import *
from python.PPOAgent import *
from python.hyperParams import *






if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("-a", "--algorithm", type=str, default="3DQN")

    args = parser.parse_args()

    cuda=False
    args.cuda = args.cuda and torch.cuda.is_available()
    print("cuda:", args.cuda)
    if(args.cuda):
        print(torch.cuda.get_device_name(0))

    env = gym.make(module) #gym env

    if(args.algorithm == "DQN"):
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams)
    elif(args.algorithm == "3DQN"):
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, double=True, duelling=True)
    elif(args.algorithm == "TD3"):
        hyperParams = TD3HyperParams()
        agent = TD3Agent(env.observation_space, env.action_space, hyperParams, cuda=args.cuda)
    elif(args.algorithm == "PPO"):
        hyperParams = PPOHyperParams()
        agent = PPOAgent(env.observation_space, env.action_space, hyperParams, continuous_action_space=isinstance(env.action_space, gym.spaces.box.Box))

    '''if("Continuous" in module): #agents are not the same wether the action space is continuous or discrete
        hyperParams = TD3HyperParams()     
        agent = TD3Agent(env.action_space, env.observation_space, hyperParams, cuda=args.cuda)
    else:
        hyperParams = DQNHyperParams()
        agent = DQNAgent(env.action_space, env.observation_space, hyperParams, cuda=args.cuda)'''

    tab_sum_rewards = []
    tab_mean_rewards = []
    for e in range(1, hyperParams.EPISODE_COUNT):
        if(args.algorithm == "PPO"):
            agent.start_episode()
        ob = env.reset()[0]
        sum_rewards=0
        steps=0
        while True:
            '''if((e-1)%(hyperParams.EPISODE_COUNT//10) == 0):
                env.render()'''
            ob_prec = ob   
            action, infos = agent.act(ob)
            ob, reward, done, _, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done, infos)
            sum_rewards += reward
            if(args.algorithm != "PPO" and steps%hyperParams.LEARN_EVERY == 0 and len(agent.buffer) > hyperParams.LEARNING_START):
                agent.learn()
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                if(args.algorithm == "PPO"):
                    agent.end_episode()
                    if(e > 0 and e%hyperParams.NUM_EP_ENV == 0):
                        agent.learn()

                tab_sum_rewards.append(sum_rewards)   
                tab_mean_rewards.append(np.mean(tab_sum_rewards[-100:]))   
                break

        print("\rEp: {} Average of last 100: {:.2f}".format(e, tab_mean_rewards[-1]), end="")
          

    
    #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
    plt.figure(figsize=(25, 12), dpi=80)
    plt.plot(tab_sum_rewards, linewidth=1)
    plt.plot(tab_mean_rewards, linewidth=1)
    plt.ylabel('Sum of the rewards')       
    plt.savefig("./images/"+module+"_"+args.algorithm+".png")
    
    #save the neural networks of the agent
    print("Saving...")
    #torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'_target.n')
    torch.save(agent.actor.state_dict(), "./trained_networks/"+module+"_"+args.algorithm+".n")

    #save the hyper parameters (for the tests and just in case)
    with open("./trained_networks/"+module+"_"+args.algorithm+".hp", 'wb') as outfile:
        pickle.dump(hyperParams, outfile)

    # Close the env (only useful for the gym envs for now)
    env.close()