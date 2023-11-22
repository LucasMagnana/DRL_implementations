import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import copy


def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def discount_rewards(rewards, list_done, gamma):
    r = []
    for reward, done in zip(reversed(rewards), reversed(list_done)):
        if done:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        r.insert(0, discounted_reward)
    r = np.array(r, dtype=np.single)
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



def compute_modifications(img1, img2):
    img_modifs = torch.flatten(img1)-torch.flatten(img2)
    modifs_index = torch.nonzero(img_modifs).squeeze()
    return (modifs_index, torch.index_select(img_modifs, 0, modifs_index))

def modify_image(img, modifs):
    img_modified = torch.flatten(img)
    img_modified[modifs[0][:]] -= modifs[1]
    return img_modified.reshape(img.shape)

    

img1 = torch.rand(4, 2, 2)
img2 = copy.deepcopy(img1)

dict_modifs = {}
modifs = compute_modifications(img1, img2)

img3 = modify_image(img1, modifs)


