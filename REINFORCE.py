import argparse
import sys
import gym 
import pickle
import torch
import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from python.NeuralNetworks import REINFORCE_Model
from python.hyperParams import REINFORCEHyperParams, module


def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r #- r.mean()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if(args.test):
        env = gym.make(module, render_mode="human") #gym env
    else:
        env = gym.make(module) #gym env


    hyperParams = REINFORCEHyperParams()

    if(args.test):
        with open('./trained_networks/'+module+'_REINFORCE.hp', 'rb') as infile:
            hyperParams = pickle.load(infile)
            


    policy = REINFORCE_Model(env.observation_space.shape[0], env.action_space.n, hyperParams)
    if(args.test):
        policy.load_state_dict(torch.load('./trained_networks/'+module+'_REINFORCE.n'))
        policy.eval()
    
    # Set up lists to hold results
    total_rewards = []
    avg_rewards = []


    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    
    # Define optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=hyperParams.LR)
    
    action_space = np.arange(env.action_space.n)
    ep = 0
    if(args.test):
        N=1
    else:
        N=hyperParams.EPISODE_COUNT
    while ep < N:
        ob_prec = env.reset()[0]
        states = []
        rewards = []
        actions = []
        done = False
        step = 0
        while not done:
            # Get actions and convert to numpy array
            action_probs = policy(torch.tensor(ob_prec)).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            ob, r, done, _, _ = env.step(action)

            states.append(ob_prec)
            rewards.append(r)
            actions.append(action)

            ob_prec = ob

            if(step == hyperParams.MAX_STEPS):
                done = True

            step += 1
            
            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards, hyperParams.GAMMA))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == hyperParams.BATCH_SIZE:
                    optimizer.zero_grad()
                    state_tensor = torch.tensor(np.array(batch_states))
                    reward_tensor = torch.tensor(batch_rewards)
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    action_tensor = action_tensor.long()
                    
                    # Calculate loss
                    logprob = torch.log(policy(state_tensor))
                    selected_logprobs = reward_tensor * torch.index_select(logprob, 1, action_tensor).diag()
                    loss = -selected_logprobs.mean()
                    #print(loss.item())
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                avg_r = np.mean(total_rewards[-100:])
                avg_rewards.append(avg_r)
                # Print running average
                ep += 1
        if(not args.test):
            print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, avg_r), end="")

    env.close()

    if(args.test):
        print("Sum of rewards:", total_rewards[-1])
    else:
        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure()
        plt.plot(total_rewards, alpha=0.75)
        plt.plot(avg_rewards, color="darkblue")
        plt.ylabel('Sum of rewards')       
        plt.savefig("./images/"+module+"_REINFORCE.png")
        
        #save the neural networks of the policy
        torch.save(policy.state_dict(), './trained_networks/'+module+'_REINFORCE.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/'+module+'_REINFORCE.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
                