import argparse
import sys
import gym 
import pickle

from multiprocessing import Pool, Queue

import threading

import matplotlib.pyplot as plt

import datetime as dt

from python.NeuralNetworks import PPO_Actor, PPO_Critic
from python.TD3Agent import *
from python.DDQNAgent import *
from python.hyperParams import PPOHyperParams, module

from test import test

def launch_agent(env, hyperParams, actor, critic, testing):
    hyperParams = hyperParams

    batch_rewards = []
    batch_advantages = []
    batch_states = []
    batch_values = []
    batch_actions = []
    batch_done = []
    total_rewards = []

    actor = actor
    critic = critic

    for y in range(hyperParams.NUM_EP_ENV):
        ob_prec = env.reset()
        states = []
        rewards = []
        actions = []
        values = []
        list_done = []
        done = False
        step=1
        while(not done and step<hyperParams.MAX_STEPS):
            if(testing):
                env.render()
            # Get actions and convert to numpy array
            action_probs = actor(torch.tensor(ob_prec)).detach().numpy()
            val = critic(torch.tensor(ob_prec)).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            ob, r, done, _ = env.step(action)

            states.append(ob_prec)
            values.extend(val)
            rewards.append(r)
            actions.append(action)
            list_done.append(done)  

            ob_prec = ob
            step+=1                

        batch_rewards.extend(discount_rewards(rewards, hyperParams.GAMMA))
        gaes = gae(np.expand_dims(np.array(rewards), 0), np.expand_dims(np.array(values), 0), np.expand_dims(np.array([not elem for elem in list_done]), 0), hyperParams.GAMMA, hyperParams.LAMBDA)
        batch_advantages.extend(gaes[0])
        batch_states.extend(states)
        batch_values.extend(values)
        batch_actions.extend(actions)
        batch_done.extend(list_done)
        total_rewards.append(sum(rewards))

    return batch_rewards, batch_advantages, batch_states, batch_values, batch_actions, batch_done, total_rewards


def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def gae(rewards, values, episode_ends, gamma, lam):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages



if __name__ == '__main__':
    env = gym.make(module) #gym env

    list_envs = []
    testing = False

    hyperParams = PPOHyperParams()

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "--test"):
            testing = True
            with open('./trained_networks/'+module+'_PPO.hp', 'rb') as infile:
                hyperParams = pickle.load(infile)

    old_actor = PPO_Actor(env.observation_space.shape[0], env.action_space.n, hyperParams)
    critic = PPO_Critic(env.observation_space.shape[0], hyperParams)

    if(testing):
        old_actor.load_state_dict(torch.load('./trained_networks/'+module+'_ac_PPO.n'))
        old_actor.eval()
        critic.load_state_dict(torch.load('./trained_networks/'+module+'_cr_PPO.n'))
        critic.eval()
        hyperParams.EPISODE_COUNT=1
        hyperParams.NUM_AGENTS=1
        hyperParams.K=0
        hyperParams.NUM_EP_ENV=1
    else:
        actor = copy.deepcopy(old_actor)   
        # Define optimizer
        optimizer_actor = torch.optim.Adam(actor.parameters(), lr=hyperParams.LR)
        optimizer_critic = torch.optim.Adam(critic.parameters(), lr=hyperParams.LR)

    mse = torch.nn.MSELoss()
    
    action_space = np.arange(env.action_space.n)

    total_rewards = []
    avg_rewards = []

    for ep in range(hyperParams.EPISODE_COUNT):
        # Set up lists to hold results
        batch_advantages = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_done = []
        batch_values = []

        #pool = Pool()
        work = []
        results = []
        for a in range(hyperParams.NUM_AGENTS):
            if(len(list_envs) < hyperParams.NUM_AGENTS):
                list_envs.append(gym.make(module))
            results.append(launch_agent(list_envs[a], hyperParams, copy.deepcopy(old_actor), copy.deepcopy(critic), testing))

        for i in range(hyperParams.NUM_AGENTS):
            ret = results[i]
            batch_rewards.extend(ret[0])
            batch_advantages.extend(ret[1])
            batch_states.extend(ret[2])
            batch_values.extend(ret[3])
            batch_actions.extend(ret[4])
            batch_done.extend(ret[5])
            total_rewards.extend(ret[6])

        for k in range(hyperParams.K):
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            state_tensor = torch.tensor(batch_states)
            advantages_tensor = torch.tensor(batch_advantages)

            values_tensor = torch.tensor(batch_values)
            rewards_tensor = torch.tensor(batch_rewards, requires_grad = True)
            rewards_tensor = rewards_tensor.float()
            # Actions are used as indices, must be 
            # LongTensor
            action_tensor = torch.LongTensor(batch_actions)
            action_tensor = action_tensor.long()

            
            # Calculate actor loss
            probs = actor(state_tensor)
            selected_probs = torch.index_select(probs, 1, action_tensor).diag()

            old_probs = old_actor(state_tensor)
            selected_old_probs = torch.index_select(old_probs, 1, action_tensor).diag()

            loss = selected_probs/selected_old_probs*advantages_tensor
            clipped_loss = torch.clamp(selected_probs/selected_old_probs, 1-hyperParams.EPSILON, 1+hyperParams.EPSILON)*advantages_tensor

            loss_actor = -torch.min(loss, clipped_loss).mean()



            # Calculate gradients
            loss_actor.backward()
            # Apply gradients
            optimizer_actor.step()

            # Calculate critic loss
            loss_critic = mse(values_tensor, rewards_tensor)  
            # Calculate gradients
            loss_critic.backward()
            # Apply gradients
            optimizer_critic.step()

        
        if(not testing):
            old_actor = copy.deepcopy(actor)
            ar = np.mean(total_rewards[-100:])
            avg_rewards.append(ar)
        # Print running average
            print("\rEp: {} Average of last 100: {:.2f}".format(ep + 1, ar), end="")
    print()

    if(testing):
        print("Sum of rewards:", total_rewards[-1])
    else:
        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure(figsize=(25, 12), dpi=80)
        plt.plot(total_rewards, linewidth=1)
        plt.plot(avg_rewards, linewidth=1)
        plt.ylabel('Sum of the rewards')       
        plt.savefig("./images/"+module+"_PPO.png")
        
        #save the neural networks of the policy
        torch.save(old_actor.state_dict(), './trained_networks/'+module+'_ac_PPO.n')
        torch.save(critic.state_dict(), './trained_networks/'+module+'_cr_PPO.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/'+module+'_PPO.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)
                

                