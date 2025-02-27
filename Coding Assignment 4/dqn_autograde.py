import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        
        # YOUR CODE HERE
        x = F.relu(self.l1(x))  
        x = self.l2(x) 

        return x

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        # YOUR CODE HERE
        self.memory.append(transition)

    def sample(self, batch_size):
        # YOUR CODE HERE
        random_batch = random.sample(self.memory, batch_size)
        return random_batch

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    
    # YOUR CODE HERE
    if it < 1000:
        epsilon = 1.0 - (0.95 * it / 1000)  
    else:
        epsilon = 0.05
    
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        # YOUR CODE HERE
        rand = random.random()
        if rand < self.epsilon: #take random action
            action = env.action_space.sample()
        else:
            with torch.no_grad():  
                q_values = self.Q(obs)          
                action = torch.argmax(q_values).item()
        
        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        # YOUR CODE HERE
        rand = random.random()
        if rand < self.epsilon: #take random action
            action = env.action_space.sample()
            print('random')
            print(action)
            print(type(action))
        else:
            with torch.no_grad():  
                q_values = self.Q(obs)          
                action = torch.argmax(q_values).item()
                print('not random')
                print(action)
                print(type(action))
        
        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        # YOUR CODE HERE
        state = torch.from_numpy(obs).type(torch.FloatTensor)

        rand = random.random()
        if rand < self.epsilon: #take random action
            action = env.action_space.sample()
            # print('random')
            # print(action)
            # print(type(action))
        else:
            with torch.no_grad():  
                q_values = self.Q(state)          
                action = torch.argmax(q_values).item()
                # print('not random')
                # print(action)
                # print(type(action))
        
        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        # YOUR CODE HERE
        state = torch.from_numpy(obs).type(torch.FloatTensor)

        rand = random.random()
        if rand < self.epsilon: #take random action
            action = env.action_space.sample()
        else:
            with torch.no_grad():  
                q_values = self.Q(state)          
                action = torch.argmax(q_values).item()
                
        
        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
