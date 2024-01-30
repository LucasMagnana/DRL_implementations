from python.TD3Agent import *
from python.DQNAgent import *
from python.PPOAgent import PPOAgent
from python.hyperParams import *

import pickle
import gym

'''class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)'''


def load_hp(args):
    if("ALE" in args.module):
        cnn = True
    else:
        cnn = False

    if(args.algorithm == "DQN"):
            hyperParams = DQNHyperParams()
    elif(args.algorithm == "3DQN"):
        if(cnn):
            hyperParams = DQNCNNHyperParams()
        else:
            hyperParams = DQNHyperParams()
    elif(args.algorithm == "TD3"):
        hyperParams = TD3HyperParams()
    elif(args.algorithm == "PPO"):
        if(cnn):
            hyperParams = PPOCNNHyperParams()
        else:
            hyperParams = PPOHyperParams()

    return hyperParams

def load_agent(args, env, continuous_action_space, hyperParams, actor_to_load=None):
    if("ALE" in args.module):
        cnn = True
    else:
        cnn = False
    
    if(actor_to_load == None and hyperParams.NUM_ENV > 1):
        ob_space = env.single_observation_space
        ac_space = env.single_action_space
    else:
        ob_space = env.observation_space
        ac_space = env.action_space
    

    if(args.algorithm == "DQN"):
            agent = DQNAgent(ob_space, ac_space, hyperParams, actor_to_load=actor_to_load, cnn=cnn)
    elif(args.algorithm == "3DQN"):
        agent = DQNAgent(ob_space, ac_space, hyperParams, double=True, duelling=True, actor_to_load=actor_to_load, cnn=cnn)
    elif(args.algorithm == "TD3"):
        agent = TD3Agent(ob_space, ac_space, hyperParams, cuda=args.cuda, actor_to_load=actor_to_load)
    elif(args.algorithm == "PPO"):
        agent = PPOAgent(ob_space, ac_space, hyperParams, continuous_action_space=continuous_action_space,\
                actor_to_load=actor_to_load, cnn=cnn)

    return agent


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

    if(isinstance(agent, PPOAgent)):
        plt.clf()
        plt.plot(agent.v_loss)
        plt.savefig("./value_loss.png")

        plt.clf()
        plt.plot(agent.e_loss)
        plt.savefig("./entropy_loss.png")

        plt.clf()
        plt.plot(agent.p_loss)
        plt.savefig("./policy_loss.png")

        plt.clf()
        plt.plot(agent.lr)
        plt.savefig("./lr.png")

    
    if(save_nn):
        #save the neural networks of the agent
        print("Saving...")
        #torch.save(agent.actor_target.state_dict(), './files/'+module+'_target.n')
        torch.save(agent.actor.state_dict(), "./files/"+module+"_"+args.algorithm+".n")

        #save the hyper parameters (for the tests and just in case)
        with open("./files/"+module+"_"+args.algorithm+".hp", 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
  