import argparse
import sys
import gym 
import pickle

import matplotlib.pyplot as plt

import datetime as dt

from python.NeuralNetworks import REINFORCE_Model
from python.TD3Agent import *
from python.DDQNAgent import *
from python.hyperParams import hyperParams, module

from test import test


def discount_rewards(rewards, gamma=hyperParams.GAMMA):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r #- r.mean()


if __name__ == '__main__':
    env = gym.make(module) #gym env

    testing = False

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "--test"):
            testing = True


    policy = REINFORCE_Model(env.observation_space.shape[0], env.action_space.n)
    if(testing):
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
    if(testing):
        N=1
    else:
        N=hyperParams.EPISODE_COUNT
    while ep < N:
        ob_prec = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            if(testing):
                env.render()
            # Get actions and convert to numpy array
            action_probs = policy(torch.tensor(ob_prec)).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            ob, r, done, _ = env.step(action)

            states.append(ob_prec)
            rewards.append(r)
            actions.append(action)

            ob_prec = ob
            
            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == hyperParams.BATCH_SIZE:
                    optimizer.zero_grad()
                    state_tensor = torch.tensor(batch_states)
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
        if(not testing):
            print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, avg_r), end="")
            if(avg_r>450):
                break

    env.close()

    if(testing):
        print("Sum of rewards:", total_rewards[-1])
    else:
        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure(figsize=(25, 12), dpi=80)
        plt.plot(total_rewards, linewidth=1)
        plt.plot(avg_rewards, linewidth=1)
        plt.ylabel('Sum of the rewards')       
        plt.savefig("./images/"+module+"_REINFORCE.png")
        
        #save the neural networks of the policy
        torch.save(policy.state_dict(), './trained_networks/'+module+'_REINFORCE.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/'+module+'_REINFORCE.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
                