import warnings
warnings.filterwarnings("ignore")
import math,time,random,getopt,sys,os,gc,psutil,itertools
from collections import namedtuple, deque
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import gym
from gym import ObservationWrapper
from gym.wrappers.monitoring.video_recorder import VideoRecorder


# Inspirations : 
#https://marl-ieee-nitk.github.io/deep-reinforcement-learning/2019/01/06/DRQN.html
#https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1/blob/main/DRQN.py
#https://arxiv.org/abs/1507.06527



##### CONSTANTS #####
SEED = 1 # To control randomness (DO NOT MODIFY)
ENV_NAME = "ParkourInfinit-v0"
# FOR TRAINING
NB_EPISODES = 1000 # Number of episodes for training
TRAINING_FREQ = 4 # How often to perform a training step
RENDER_MODE_TRAIN = False # Set to True to render simulation during training
MAX_TIMESTEPS = 5000 # Naximum number of timesteps in one episode (max is 5000 in Gym env)
# FOR TESTING
RENDER_MODE_TEST = False # Set to True to render simulation during testing
RECORD_ALL = True # True to record all tests, False to record last test
# Hyper parameters 
MEMORY_SIZE = 50 # Size of the Replay Memory of the agent (number of episodes to store)
SEQUENCE_SIZE = 4 # Number of sequential experiences to use for RNN
EXPLORATION_RATE = 1 # Randomness of the agent's action 
EXPLORATION_DECAY_RATE = 0.999999 # To reduce exploration rate with exploration decay method
MIN_EXPLORATION_RATE = 0.001 # Minimum exploration rate
BATCH_SIZE = 8 # Training batch size (by episode, real batch size / sequence size should be OK)
LEARNING_RATE = 0.0001 #0.00025 # DNN optimizer learning rate
GAMMA = 0.99 # DQN discount factor
TARGET_NN_UPDATE = 10000 ##NOT USED## # Update target network after X learned examples (method 1) 
TARGET_NN_UR = 0.001 #0.01 # Update target network with an update rate (method 2)

# To load tensors on GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class CustomObservation(ObservationWrapper):
    '''
    Observation wrapper that modify the observation to get POV Box.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['pov']

    def observation(self, observation):
        return np.array(observation['pov'])



class ReplayMemory(object):
    '''
    Sequential memory of the DRQN.
    '''
    def __init__(self, mem_size=MEMORY_SIZE, seq_size=SEQUENCE_SIZE):
        self.mem_size = mem_size # Maximum number of episodes to store
        self.seq_size = seq_size # Size of the sequence for sampling (number of experiences) 
        # Memory is a deque storing the last mem_size episodes
        # For each episode, it stores the experiences encountered in sequential order
        self.memory = deque(maxlen=mem_size)

    def push(self, episode):
        self.memory.append(episode)

    def sample(self, batch_size):
        assert len(self.memory) >= batch_size
        samples = []
        # Sample batch_size episodes from memory (except current one)
        episodes = random.sample(self.memory, batch_size)
        # Minimum number of experiences in episodes
        min_seq = min([len(ep) for ep in episodes])
        # Determine sequence size to use
        seq_size = self.seq_size if min_seq > self.seq_size else min_seq
        # Get a sequence from each sampled episode
        for episode in episodes:
            idx = random.randint(0, len(episode)-seq_size)
            sample = episode.sample(seq_size, idx)
            samples.append(sample)
        return samples, seq_size

    def __len__(self):
        return len(self.memory)



class EpisodeBuffer:
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []

    def put(self, transition):
        self.state.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_state.append(transition[3])
        self.done.append(transition[4])

    def sample(self, seq_size, idx):
        state = self.state[idx:idx+seq_size]
        action = self.action[idx:idx+seq_size]
        reward = self.reward[idx:idx+seq_size]
        next_state = self.next_state[idx:idx+seq_size]
        done = self.done[idx:idx+seq_size]
        return dict(state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done)

    def __len__(self):
        return len(self.state)



class DRQN(nn.Module):
    '''
    Deep Recurrent Q-Network with CNN and RNN.
    '''
    # action_space is output size
    def __init__(self, input_channels, action_space):
        super().__init__()
        # Pre-trained model with transfer learning
        #self.cnn = models.resnet18(pretrained=True)
        # Finetuning 
        #for param in self.cnn.parameters():
        #    param.requires_grad = False
        #fc_input_size = self.cnn.fc.in_features
        #self.cnn.fc = nn.Identity() # No output

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        fc_input_size = 3136

        # RNN
        self.rnn_layers = 1
        self.rnn_out = 512
        self.rnn = nn.LSTM(input_size=fc_input_size, hidden_size=self.rnn_out, num_layers=self.rnn_layers, batch_first=True)
        
        # Output layer
        self.out = nn.Linear(self.rnn_out, action_space)
    
    def forward(self, x, hidden):
        '''Forward propagation'''
        x = x.float() # state is stored as int8 to save memory
        state_shape = x.shape # (batch,seq,channels,h,w)
        # Join batch and seq to compute convolutions
        x = x.view(-1, state_shape[2], state_shape[3], state_shape[4])
        # CNN 
        x = self.cnn(x)
        # Change shape to (batch, seq, out)
        x = x.view(state_shape[0], state_shape[1], -1)
        # RNN
        x, new_hidden = self.rnn(x,hidden)
        # Output layer
        return self.out(x), new_hidden

    def init_hidden(self, batch_size=1, device=torch.device("cpu")):
        return (torch.zeros(self.rnn_layers, batch_size, self.rnn_out, device=device), torch.zeros(self.rnn_layers, batch_size, self.rnn_out, device=device))



class DRQNAgent():
    '''
    Deep Recurrent Q-Learning Agent.
    '''
    def __init__(self, state_space, action_space, input_channels, 
            memory_size=MEMORY_SIZE,
            sequence_size=SEQUENCE_SIZE,
            exploration_rate=EXPLORATION_RATE,
            decrease_rate=EXPLORATION_DECAY_RATE,
            min_exploration_rate=MIN_EXPLORATION_RATE,
            training_batch_size=BATCH_SIZE,
            model_learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            target_nn_update=TARGET_NN_UPDATE,
            target_nn_update_rate=TARGET_NN_UR,
            is_training = True,
            device=torch.device("cpu")):
        
        # Device to use (GPU or CPU)
        self.device = device

        # Network input and output size
        self.state_space = state_space
        self.action_space = action_space
        
        # Model and Target Networks
        self.is_training = is_training
        self.model = DRQN(input_channels, action_space).to(device)
        self.target_model = DRQN(input_channels, action_space).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target model is used in evaluation mode
        # Hyper-parameters
        self.batch_size = training_batch_size
        self.gamma = gamma
        # Optimizer
        self.criterion = nn.HuberLoss() #nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_learning_rate)
        # To update target network
        self.nb_train = 0
        self.target_update = target_nn_update
        self.target_ur = target_nn_update_rate
        
        # Replay memory
        # Memory size is the number of episodes and sequence is the size of the sequence for LSTM
        self.memory = ReplayMemory(memory_size, sequence_size)

        # Exploration
        self.eps = exploration_rate
        self.dcr = decrease_rate
        self.min_eps = min_exploration_rate


    def save_model(self, path='model.pth', with_optimizer=False, epoch=None):
        '''Save DNN model to a file'''
        if with_optimizer: # To keep training later
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
        else:
            torch.save({
                'model': self.model.state_dict()
            }, path)


    def load_model(self, path='model.pth', with_optimizer=False):
        '''Load DNN model from a file'''
        save = torch.load(path)
        self.model.load_state_dict(save['model'])
        self.target_model.load_state_dict(save['model'])
        if with_optimizer:
            self.optimizer.load_state_dict(save['optimizer'])


    def set_eval_mode(self):
        '''Set agent to evaluation mode'''
        self.is_training = False
        self.model.eval()


    def set_train_mode(self):
        '''Set agent to training mode'''
        self.is_training = True
        self.model.train()


    def preprocess(self, input_state):
        '''
        Preprocess input state.

        Takes a list/ndarray/Tensor of shape : (height, width, channels)
        Output shape : (batch, channels, height, width)
        '''
        # Map input to tensor
        if not isinstance(input_state, np.ndarray):
            input_state = np.array(input_state)
        if not(torch.is_tensor(input_state)):
            input_state = torch.from_numpy(input_state.copy()).float().to(self.device)
        # Change dim from (height, width, channel) to (channel, height, width)
        input_state = input_state.permute(2,0,1)
        # Separate tensor into two : colors / depth
        input_state_colors = input_state[0:3,:,:] 
        input_state_depth = input_state[3,:,:].unsqueeze(0)
        # Set of transformations on the input state
        # ToTensor -> T.ToPILImage is Trick to normalize between 0 & 1
        def transforms(img, to_grayscale=False):
            img = T.Resize(self.state_space)(img)
            # Can not normalize if storing as uint8
            # Still works with though but I imagine with lose a lot of info
            #TODO
            #img = T.ToPILImage()(img)
            #img = T.ToTensor()(img)
            #if img.shape[0] == 3: # For colored images
            #    img = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            #else: # Else compute it
            #    mean, std = img.mean([1,2]), img.std([1,2])
            #    img = T.Normalize(mean,std)(img)
            if to_grayscale:
                img = T.Grayscale()(img)
            return img.to(self.device)
        # Apply transformations
        input_state_tr = transforms(input_state_colors, to_grayscale=True)
        input_state_depth = transforms(input_state_depth)
        # Concat channels
        input_state = torch.cat((input_state_tr,input_state_depth), dim=0)
        # Add one dimension for batch & seq -> (batch, seq, channels, height, width)
        # Convert it to uint8 to reduce memory usage
        input_state = input_state.unsqueeze(0).unsqueeze(0).to(torch.uint8)
        return input_state


    def select_action(self, state, hidden):
        '''Choose the best action according to DNN prediction'''
        # Use model to predict best Q-values 
        with torch.no_grad():
            out, new_hidden = self.model(state, hidden)
        # Random between 0 and 1
        rand = random.random()
        #Exploration with decay (set min_eps = eps for e-greedy exploration)
        if self.is_training:
            self.eps = self.eps * self.dcr
            self.eps = max(self.eps, self.min_eps) #Minimum
        # Best action 
        if not(self.is_training) or rand > self.eps:
            return torch.argmax(out).item(), new_hidden
        # Random action
        else:
            return random.randrange(self.action_space), new_hidden


    def train(self):
        '''One training step using agent's memory'''
        # Make sure the memory is big enough (must contains batch_size episodes)
        if len(self.memory) <= self.batch_size:
            return

        # Counter to update target network
        #self.nb_train += self.batch_size

        # Get random sequences from episodes in memory
        samples, seq_len = self.memory.sample(self.batch_size)

        # Modify structures 
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for i in range(self.batch_size):
            states.append(torch.cat(samples[i]["state"],dim=1))
            actions.append(samples[i]["action"])
            rewards.append(samples[i]["reward"])
            next_states.append(torch.cat(samples[i]["next_state"],dim=1))
            dones.append(samples[i]["done"])
        in_shape = (self.batch_size,seq_len) + (states[0].shape[2:])
        states = torch.ByteTensor(torch.cat(states).view(in_shape)).to(self.device) #iunt8
        actions = torch.LongTensor(np.array(actions).reshape(self.batch_size,seq_len,-1)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(self.batch_size,seq_len,-1)).to(self.device)
        next_states = torch.ByteTensor(torch.cat(next_states).view(in_shape)).to(self.device) #uint8
        dones = torch.FloatTensor(np.array(dones).reshape(self.batch_size,seq_len,-1)).to(self.device)
        
        # Compute Q-values and keep the ones associated to the stored actions
        hidden = self.model.init_hidden(self.batch_size, device=self.device)
        q_pred = self.model(states, hidden)[0].gather(2, actions)

        # Initialize Q-values of next states to 0 and compute values for non final ones
        hidden = self.model.init_hidden(self.batch_size, device=self.device)
        q_next = self.target_model(next_states, hidden)[0].max(2)[0].detach() #Get max from pred

        # Compute loss
        q_target = (rewards + self.gamma * q_next.unsqueeze(2)*dones)
        loss = self.criterion(q_pred,q_target).to(self.device)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        # Hard method
        #if self.nb_train > self.target_update:
        #    self.target_model.load_state_dict(self.model.state_dict()) 
        #    self.nb_train = 0
        # Soft method
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.target_ur*model_param.data + (1.0 - self.target_ur)*target_param.data)




# Load environment
env = gym.make(ENV_NAME)
# Transform observation to observation['pov']
# In this case, we need only the POV observation
env = CustomObservation(env)


# List of available actions
#Single actions (without 'back' as not needed)
actions = ["forward", "left", "right", "jump", "sprint"]
# Generates combinations of actions of 2 and 3 elements
action_keys = list(itertools.combinations(actions, 2))
action_keys += list(itertools.combinations(actions, 3))
# Add single actions (except jump)
action_keys += [(action,) for action in actions if action != 'jump']
# Remove actions that can stuck Agent left+right
action_keys = [a for a in action_keys if not ('left' in a and 'right' in a)]
# => 21 available action


#Size of desired input image
state_space = (84,84)


#Number of available actions
action_space = len(action_keys)


def ind_to_action(ind):
    keys = action_keys[ind] # Get the action combination from index
    action = env.action_space.noop() # Init an action dict
    action['camera'] = 0,0 # Needed for rendering
    for key in keys:
        # We need to process camera action apart
        # /!\ Not implemented at the moment /!\ 
        if "camera" in key:
            action["camera"] = 0,0#1,0 if key.split("_")[1] == "x" else 0,1
        else:
            action[key] = 1 # Set to 1 as it is binary activation
    return action


# Dynamically plot training evolution
def plot(interactions,rewards):
    plt.clf() # Clear
    plt.title("Evolution du training")
    plt.plot(interactions, rewards)
    plt.ylabel('Cumul des rewards')
    plt.xlabel('Nb interactions')
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.pause(0.001) # to update info we need to pause a little bit
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# To control the randomness
np.random.seed(SEED)
env.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = True # ERROR sometimes ... Check CUDNN_STATUS_EXECUTION_FAILED
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# For debugging purposes
def show_RAM_usage():
    py = psutil.Process(os.getpid())
    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))



def train():
    '''Train an agent on ParkourInfinit-v0 environment'''
    # DQN Agent 
    agent = DRQNAgent(state_space, action_space, input_channels=2, device=device)

    # To evaluate the training
    rewards = []
    interactions = []
    nb_interactions = 0

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    
    # Run the simulation NB_EPISODES times
    for i_episode in range(NB_EPISODES):
    
        # Reset the environment
        obs = env.reset()

        # Preprocess state image
        hidden = agent.model.init_hidden(1, device=device)
        state = agent.preprocess(obs)

        # Total of reward during the episode 
        tot_ep_reward = 0

        # Record experiences for RNN
        episode_record = EpisodeBuffer()
    
        # Run the actions of the agent
        for t in range(MAX_TIMESTEPS):

            # Display
            if RENDER_MODE_TRAIN:
                env.render()
        
            # The agent selects the action and the environment performs it
            action_ind, hidden = agent.select_action(state, hidden)
            next_obs, reward, done, info = env.step(ind_to_action(action_ind))

            # Preprocess next state image
            next_state = agent.preprocess(next_obs)

            # Store the transition in memory (We store in CPU to prevent memory leak)
            episode_record.put([state.cpu(), action_ind, reward, next_state.cpu(), (0.0 if done else 1.0)])

            # Update state
            state = next_state

            # Update info to evaluate training
            nb_interactions += 1
            tot_ep_reward += int(reward)

            # Training
            if t % TRAINING_FREQ == 0:
                agent.train()

            if done:
                break

        # Save episode in memory
        agent.memory.push(episode_record)
        
        # Result of the episode
        print(f"Episode {i_episode}, score:{tot_ep_reward}, exploration_r: {round(agent.eps,4)}")

        # Save result of an episode to evaluate training
        rewards.append(tot_ep_reward)
        interactions.append(nb_interactions)

        # Interactive plotting
        plot(interactions,rewards)

        # Save model each 10 episodes
        if i_episode % 10 == 0:
            agent.save_model(path=f'model_save.pth')
            torch.cuda.empty_cache() # Should not be used but to prevent memory leak

    # Save last perfs
    agent.save_model(path=f'model.pth')

    # Stop dynamic plotting and show final graph
    plt.ioff()
    plt.show()
    plt.title("Evolution du training")
    plt.ylabel('Cumul des rewards')
    plt.xlabel('Nb interactions')
    plt.plot(interactions, rewards)
    plt.savefig(f'res/score.png')

    # Custom env error, not our fault :(
    try:
        env.close()
    except Exception as e:
        pass 



def test(nb_episodes):
    '''Test an agent on ParkourInfinit-v0 environment'''
    # Please see train function for explanation of the code
    video_recorder = VideoRecorder(env=env,path="res/demo.mp4")
    agent = DRQNAgent(state_space, action_space, input_channels=2, device=device)
    agent.load_model(path=f'model.pth')
    agent.set_eval_mode()
    tot_reward = 0
    rewards = []
    for i_episode in range(nb_episodes):
        obs = env.reset()
        hidden = agent.model.init_hidden(1, device=device)
        state = agent.preprocess(obs)
        tot_ep_reward = 0
        done = False
        while not done:
            if RENDER_MODE_TEST:
                env.render()            
            if RECORD_ALL or i_episode+1 == nb_episodes:
                video_recorder.capture_frame()
            action_ind, hidden = agent.select_action(state, hidden)
            next_obs, reward, done, info = env.step(ind_to_action(action_ind))
            state = agent.preprocess(next_obs)
            action = action_ind
            tot_ep_reward += reward
        print(f"Episode {i_episode}, score:{tot_ep_reward}")
        tot_reward += tot_ep_reward
        rewards.append(tot_ep_reward)
    print(f"Moyenne des recompenses sur {nb_episodes} episodes : {tot_reward/nb_episodes}")
    print(f"Minimum des recompenses sur {nb_episodes} episodes : {min(rewards)}")
    print(f"maximum des recompenses sur {nb_episodes} episodes : {max(rewards)}")
    try:
        env.close()
    except Exception as e:
        pass
    video_recorder.close()
    video_recorder.enabled = False






# ----------------------------------------------------------------------------- #
# ---------------------------     Main       ---------------------------------- #
# ----------------------------------------------------------------------------- #

def usage():
    print(f"Usage : <executable> [ --train OR --test=<nb_episodes> ]")

def main(argv):
    run_train = True
    test_v = 0
    try:   
        opts,_ = getopt.getopt(argv, '', [ 'train', 'test='])
        if (len(opts) == 0):
            raise ValueError()
        for (opt,val) in opts:
            if opt == "--train":
                run_train = True
                break
            elif opt == "--test" and int(val):
                run_train = False    
                test_v = int(val)
                break
            else:
                raise ValueError()
    except (getopt.GetoptError, ValueError) as e:
        print("Error : Impossible de lancer le programme...")
        usage()
        sys.exit(2)
    # We dont want functions to be in the try / except block
    if run_train:
        train()
    else:
        test(test_v)



if __name__ == '__main__':
    main(sys.argv[1:])
