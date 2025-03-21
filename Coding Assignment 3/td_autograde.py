import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        
        # YOUR CODE HERE
        n_actions = self.Q.shape[1]
        action_probability = np.ones(n_actions) * self.epsilon / n_actions
        action_probability[np.argmax(self.Q[state])] += (1 - self.epsilon)
        
        return np.random.choice(n_actions, p=action_probability)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0 
        R = 0 

        # YOUR CODE HERE
        state = env.reset()
        action = policy.sample_action(state)
        done = False
        while not done:
            new_state, reward, done, p = env.step(action)
            new_action = policy.sample_action(new_state)

            Q[state][action] += alpha * (reward + discount_factor * Q[new_state][new_action] - Q[state][action])

            state = new_state
            action = new_action

            R += reward * (discount_factor**i)
            i += 1


        stats.append((i, R))
        
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
    
        # YOUR CODE HERE
        state = env.reset()
        done = False

        while not done:
            action = policy.sample_action(state)
            new_state, reward, done, p = env.step(action)
            
            best_action = np.argmax(Q[new_state])
            Q[state][action] += alpha * (reward + discount_factor * Q[new_state][best_action] - Q[state][action])

            state = new_state
            
            R += reward * (discount_factor**i)
            i += 1
        
        stats.append((i, R))
        
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)
