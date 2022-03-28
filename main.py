import argparse
import sys
import gym 
import pickle
import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from python.TD3Agent import *
from python.DDQNAgent import *
from python.hyperParams import *

from test import test





if __name__ == '__main__':

    cuda=False
    cuda = cuda and torch.cuda.is_available()
    print("cuda:", cuda)
    if(cuda):
        print(torch.cuda.get_device_name(0))

    env = gym.make(module) #gym env

    if("Continuous" in module): #agents are not the same wether the action space is continuous or discrete
        hyperParams = TD3HyperParams()     
        agent = TD3Agent(env.action_space, env.observation_space, hyperParams, cuda=cuda)
    else:
        hyperParams = DDQNHyperParams()
        agent = DDQNAgent(env.action_space, env.observation_space, hyperParams, cuda=cuda)

    tab_sum_rewards = []
    tab_mean_rewards = []

    for e in range(1, hyperParams.EPISODE_COUNT):
        ob = env.reset()
        sum_rewards=0
        steps=0
        while True:
            if((e-1)%(hyperParams.EPISODE_COUNT//10) == 0):
                env.render()
            ob_prec = ob   
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done)
            sum_rewards += reward
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                if(len(agent.buffer)>hyperParams.LEARNING_START):
                    agent.learn(steps)
                tab_sum_rewards.append(sum_rewards)   
                tab_mean_rewards.append(np.mean(tab_sum_rewards[-100:]))   
                break

        print("\rEp: {} Average of last 100: {:.2f}".format(e, tab_mean_rewards[-1]), end="")
          

    
    #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
    plt.figure(figsize=(25, 12), dpi=80)
    plt.plot(tab_sum_rewards, linewidth=1)
    plt.plot(tab_mean_rewards, linewidth=1)
    plt.ylabel('Sum of the rewards')       
    plt.savefig("./images/"+module+".png")
    
    #save the neural networks of the agent
    print("Saving...")
    torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'_target.n')
    torch.save(agent.actor.state_dict(), './trained_networks/'+module+'.n')

    #save the hyper parameters (for the tests and just in case)
    with open('./trained_networks/'+module+'.hp', 'wb') as outfile:
        pickle.dump(hyperParams, outfile)

    # Close the env (only useful for the gym envs for now)
    env.close()