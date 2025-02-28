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

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[len(self.memory) % self.capacity] = transition

    def sample(self, batch_size):
        # YOUR CODE HERE
        random_batch = random.sample(self.memory, batch_size)
        return random_batch

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    
    ## YOUR CODE HERE
    if it < 1000:
        epsilon = 1.0 - (0.95 * it / 1000)
    else:
        epsilon = 0.05

    # epsilon = max(1.0 - (0.95 * it / 1000), 0.05)
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
        state = torch.from_numpy(obs).type(torch.FloatTensor)

        rand = random.random()
        if rand < self.epsilon: #take random action
            action = np.random.choice([0, 1])
        else:
            with torch.no_grad():  
                q_values = self.Q(state)          
                action = torch.argmax(q_values).item()

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE

    return Q(states).gather(dim=1, index=actions)
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of rewards. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """

    # YOUR CODE HERE
    next_Q = Q(next_states)
    next_Q[dones.squeeze()] = 0

    return rewards + discount_factor * next_Q.max(dim=1).values.unsqueeze(1)

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            # YOUR CODE HERE
            # set epsilon
            policy.set_epsilon(get_epsilon(global_steps))

            # sample action from policy
            action = policy.sample_action(state)

            # get next state, action, reward from policy
            s_next, reward, done, _ = env.step(action)

            # store in memory
            memory.push((state, action, reward, s_next, done))

            loss = train(Q, memory, optimizer, batch_size, discount_factor)

            global_steps += 1
            steps += 1
            state = s_next

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                break
    return episode_durations
