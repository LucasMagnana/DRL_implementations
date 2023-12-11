from python.TD3Agent import *
from python.DQNAgent import *
from python.PPOAgent import PPOAgent
from python.hyperParams import *

import pickle

def load_agent_and_hp(args, env, continuous_action_space, actor_to_load=None):
    if("ALE" in args.module):
        cnn = True
    else:
        cnn = False

    if(args.algorithm == "DQN"):
            hyperParams = DQNHyperParams()
            agent = DQNAgent(env.observation_space, env.action_space, hyperParams, actor_to_load=actor_to_load, cnn=cnn)
    elif(args.algorithm == "3DQN"):
        if(cnn):
            hyperParams = DQNCNNHyperParams()
        else:
            hyperParams = DQNHyperParams()
        agent = DQNAgent(env.observation_space, env.action_space, hyperParams, double=True, duelling=True, actor_to_load=actor_to_load, cnn=cnn)
    elif(args.algorithm == "TD3"):
        hyperParams = TD3HyperParams()
        agent = TD3Agent(env.observation_space, env.action_space, hyperParams, cuda=args.cuda, actor_to_load=actor_to_load)
    elif(args.algorithm == "PPO"):
        hyperParams = PPOHyperParams()
        agent = PPOAgent(env.observation_space, env.action_space, hyperParams, continuous_action_space=continuous_action_space,\
                actor_to_load=actor_to_load, cnn=cnn)

    return hyperParams, agent


def save(tab_sum_rewards, tab_mean_rewards, module, args, agent, hyperParams, save_nn=False):
    #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
    print()
    print("Ploting...")
    plt.close()
    plt.figure()
    plt.plot(tab_sum_rewards, alpha=0.75)
    plt.plot(tab_mean_rewards, color="darkblue")
    plt.xlabel('Episodes')   
    plt.ylabel('Sum of rewards')       
    plt.savefig("./images/"+module+"_"+args.algorithm+".png")
    
    if(save_nn):
        #save the neural networks of the agent
        print("Saving...")
        #torch.save(agent.actor_target.state_dict(), './files/'+module+'_target.n')
        torch.save(agent.actor.state_dict(), "./files/"+module+"_"+args.algorithm+".n")

        #save the hyper parameters (for the tests and just in case)
        with open("./files/"+module+"_"+args.algorithm+".hp", 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
  