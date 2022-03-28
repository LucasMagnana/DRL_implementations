#LunarLanderContinuous hyper parameters (solves it but not optimal)
class TD3HyperParams :
    def __init__(self):
        self.HIDDEN_SIZE = 400
        self.ACT_INTER = 300

        self.CRIT_IN = 400
        self.CRIT_INTER = 300

        self.EPISODE_COUNT = 5000
        self.POLICY_DELAY = 2

        self.BUFFER_SIZE = 5e5  # replay buffer size
        self.BATCH_SIZE = 100      # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 5e-3           # for soft update of target parameters
        self.LR_ACTOR = 0.001     # learning rate of the actor 
        self.LR_CRITIC = 0.001       # learning rate of the critic
        self.WEIGHT_DECAY = 0      # L2 weight decay

        self.POLICY_NOISE = 0.2
        self.NOISE_CLIP = 0.5

        self.LEARNING_START = 25*self.BATCH_SIZE
        self.MAX_STEPS = 2000
        self.EXPLORATION_NOISE = 0.1


#Cartpole hyper parameters (solves it but not optimal)
class DDQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 1e25 
        self.ALPHA = 0.05 #
        self.GAMMA = 0.99
        self.LR = 0.01
        self.BATCH_SIZE = 10

        self.HIDDEN_SIZE = 16
        self.ACT_INTER = 16

        self.EPISODE_COUNT = 3000
        self.MAX_STEPS = 1000
        self.LEARNING_START = 0

        self.EPSILON = 1.0
        self.MIN_EPSILON = 0
        self.EPSILON_DECAY = self.EPSILON/(self.EPISODE_COUNT*4/5)


class REINFORCEHyperParams :
    def __init__(self):
        self.LR = 0.01
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99

        self.HIDDEN_SIZE = 16
        self.ACT_INTER = 16

        self.EPISODE_COUNT = 3000
        self.MAX_STEPS = 1000


class PPOHyperParams :
    def __init__(self):
        self.LR = 0.01
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99
        self.LAMBDA = 0.99
        self.EPSILON = 0.2

        self.EPISODE_COUNT = 60
        self.NUM_AGENTS = 5
        self.NUM_EP_ENV = 10
        self.K = 5

        self.HIDDEN_SIZE = 16
        self.ACT_INTER = 16

        self.MAX_STEPS = 1000

module = "CartPole-v1" #"LunarLanderContinuous-v2"
