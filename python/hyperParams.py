#LunarLanderContinuous hyper parameters (solves it but not optimal)
class TD3HyperParams :
    def __init__(self):
        self.HIDDEN_SIZE_1 = 256
        self.HIDDEN_SIZE_2 = 256
        
        self.EPISODE_COUNT = 2000
        self.POLICY_DELAY = 2

        self.BUFFER_SIZE = 5e5  # replay buffer size
        self.BATCH_SIZE = 128      # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 5e-3           # for soft update of target parameters
        self.LR_ACTOR = 1e-4     # learning rate of the actor 
        self.LR_CRITIC = 1e-3     # learning rate of the critic
        self.WEIGHT_DECAY = 0      # L2 weight decay

        self.POLICY_NOISE = 0.2
        self.NOISE_CLIP = 0.5

        self.LEARN_EVERY = 4

        self.LEARNING_START = 25*self.BATCH_SIZE
        self.MAX_STEPS = 1000
        self.EXPLORATION_NOISE = 0.1


#Cartpole hyper parameters (solves it but not optimal)
class DQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 1e6
        self.TAU = 1/10000
        self.GAMMA = 0.99
        self.LR = 0.00025
        self.BATCH_SIZE = 32

        self.HIDDEN_SIZE_1 = 64
        self.HIDDEN_SIZE_2 = 64

        self.TRAINING_FRAMES = 1e7
        self.MAX_STEPS = 1e10
        self.LEARNING_START = 50000
        self.LEARN_EVERY = 1

        self.EPSILON = 1.0
        self.MIN_EPSILON = 0.1
        self.EPSILON_DECAY = self.EPSILON/(self.BUFFER_SIZE)


class REINFORCEHyperParams :
    def __init__(self):
        self.LR = 0.01
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99

        self.HIDDEN_SIZE = 256
        self.ACT_INTER = 256

        self.EPISODE_COUNT = 1000
        self.MAX_STEPS = 1000


class PPOHyperParams :
    def __init__(self):
        self.LR = 1e-4
        self.GAMMA = 0.99
        self.LAMBDA = 0.99
        self.EPSILON = 0.1

        self.HIDDEN_SIZE_1 = 256
        self.HIDDEN_SIZE_2 = 256

        self.TRAINING_FRAMES = 1e7
        self.K = 10

        self.BATCH_SIZE = 32
        self.MAXLEN = 1000

        self.MAX_STEPS = 1e10


