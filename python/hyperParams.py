#LunarLanderContinuous hyper parameters (solves it but not optimal)
class TD3HyperParams :
    def __init__(self):
        self.HIDDEN_SIZE_1 = 256
        self.HIDDEN_SIZE_2 = 256
        
        self.TRAINING_FRAMES = 3e5
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


class DQNCNNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 1e6
        self.TARGET_UPDATE = 1e4
        self.GAMMA = 0.99
        self.LR = 1e-5
        self.BATCH_SIZE = 32

        self.HIDDEN_SIZE_2 = 512

        self.TRAINING_FRAMES = 3e7
        self.MAX_STEPS = 18000
        self.LEARNING_START = 5e4
        self.LEARN_EVERY = 4
        self.NOOP = 30

        self.START_EPSILON = 1.0
        self.FIRST_MIN_EPSILON = 0.1
        self.FIRST_EPSILON_DECAY = (self.START_EPSILON-self.FIRST_MIN_EPSILON)/(1e6-self.LEARNING_START)
        self.MIN_EPSILON = 0.01
        self.SECOND_EPSILON_DECAY = (self.FIRST_MIN_EPSILON-self.MIN_EPSILON)/1e6

#CARTPOLE
class DQNHyperParams : 
    def __init__(self):
        self.BUFFER_SIZE = 1e5
        self.TARGET_UPDATE = 1e3
        self.GAMMA = 0.99
        self.LR = 2.5e-4
        self.BATCH_SIZE = 32

        self.HIDDEN_SIZE_1 = 64
        self.HIDDEN_SIZE_2 = 64

        self.TRAINING_FRAMES = 1e5
        self.MAX_STEPS = 1000
        self.LEARNING_START = 64
        self.LEARN_EVERY = 4

        self.START_EPSILON = 1.0
        self.FIRST_MIN_EPSILON = 0.1
        self.FIRST_EPSILON_DECAY = (self.START_EPSILON-self.FIRST_MIN_EPSILON)/(3e4-self.LEARNING_START)
        self.MIN_EPSILON = 0.01
        self.SECOND_EPSILON_DECAY = (self.FIRST_MIN_EPSILON-self.MIN_EPSILON)/3e4

#CARTPOLE
class PPOHyperParams : 
    def __init__(self):
        self.GAMMA = 0.99
        self.LR = 1e-4
        self.BATCH_SIZE = 256

        self.HIDDEN_SIZE_1 = 256
        self.HIDDEN_SIZE_2 = 256

        self.TRAINING_FRAMES = 5e5
        self.MAX_STEPS = 1000

        self.LAMBDA = 0.99
        self.EPSILON = 0.1

        self.K = 4
        self.MAXLEN = 1000

        self.ENTROPY_COEFF = 0.01
        self.VALUES_COEFF = 0.5
        self.MAX_GRAD = 0.5
        self.GAMMA_GAE = 0.95
        self.LR_DECAY = self.LR/(self.TRAINING_FRAMES/self.MAXLEN)



class PPOCNNHyperParams : 
    def __init__(self):
        self.GAMMA = 0.99
        self.LR = 2.5e-4
        self.BATCH_SIZE = 256

        self.HIDDEN_SIZE_1 = 512
        self.HIDDEN_SIZE_2 = 512

        self.TRAINING_FRAMES = 1e7
        self.MAX_STEPS = 4500

        self.LAMBDA = 0.99
        self.EPSILON = 0.1

        self.K = 4
        self.MAXLEN = 1024
        self.NOOP = 30

        self.ENTROPY_COEFF = 0.01
        self.VALUES_COEFF = 0.5
        self.MAX_GRAD = 0.5
        self.GAMMA_GAE = 0.95
        self.LR_DECAY = self.LR/(self.TRAINING_FRAMES/self.MAXLEN)



class REINFORCEHyperParams :
    def __init__(self):
        self.LR = 0.01
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99

        self.HIDDEN_SIZE = 256
        self.ACT_INTER = 256

        self.EPISODE_COUNT = 1000
        self.MAX_STEPS = 1000


