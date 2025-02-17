import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with less than 20 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair.

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        probs = []


        for i in range(len(states)):
            if states[i][0] >= 20:
                if actions[i] == 0: #stick
                    probs.append(1) #bigger then 20, thus stick is prob 1
                else: #hit
                    probs.append(0)
            else:
                if actions[i] == 0: #stick
                    probs.append(0) #smaller 20, thus stick is prob 0
                else: #hit
                    probs.append(1)

        # states = np.array([s[0] for s in states]) >= 20
        # _probs = states ^ np.array(actions)

        return np.array(probs)

    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        if state[0] >= 20:
            action = 0 #stick
        else:
            action = 1 #hit


        return action #0 if state[0] >= 20 else 1

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length.
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    # YOUR CODE HERE
    state = env.reset() 
    done = False

    while not done:
        states.append(state)
        action = policy.sample_action(state)
        actions.append(action)

        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        dones.append(done)

        state = next_state

    return states, actions, rewards, dones

def mc_prediction(policy, env, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A policy which allows us to sample actions with its sample_action method.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)

    # YOUR CODE HERE
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, policy)

        G = 0

        visited_states = []

        for j in reversed(range(len(states))):
            G = discount_factor * G + rewards[j]
            state = states[j]

            # first-visit monte carlo algorithm
            if state not in visited_states:
                visited_states.append(state)
                returns_count[state] += 1
                V[state] += (G - V[state]) / returns_count[state]

    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains
        a probability of perfoming action in given state for every corresponding state action pair.

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """

        # YOUR CODE HERE

        return np.ones(len(states)) * 0.5

    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            state: current state

        Returns:
            An action (int).
        """

        # YOUR CODE HERE

        return np.random.choice([0, 1])


def mc_importance_sampling(behavior_policy, target_policy, env, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.

    Args:
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)

    get_sampling_ratio = lambda s, a: target_policy.get_probs([s], [a])[0] / behavior_policy.get_probs([s], [a])[0]

    # YOUR CODE HERE
    for _ in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)

        G = 0
        W = 1

        for state, action, reward in reversed(list(zip(states, actions, rewards))):
            W *= get_sampling_ratio(state, action)
            G = discount_factor * G + reward

            returns_count[state] += 1
            V[state] += ((W * G) - V[state]) / returns_count[state]

    return V
