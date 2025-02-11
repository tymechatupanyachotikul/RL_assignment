import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE

    while True:
        delta = 0
        for s in range(env.nS):
            updated_v = 0
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    updated_v += (prob * (reward + discount_factor * V[next_state])) * policy[s][a]

            delta = max(delta, abs(V[s] - updated_v))
            V[s] = updated_v

        if delta < theta:
            break

    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        policy_stable = True

        for s in range(env.nS):
            action_value = np.zeros(env.nA)
            old_action = policy[s][:]

            for a in range(env.nA):
                for (prob, next_state, reward, done) in env.P[s][a]:
                    action_value[a] += prob * (reward + discount_factor * V[next_state])

            policy[s] = np.eye(env.nA)[np.argmax(action_value)]

            if not np.array_equal(old_action, policy[s]):
                policy_stable = False

        if policy_stable:
            break

    return policy, policy_eval_v(policy, env, discount_factor)

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """

    # Start with a all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE

    while True:
        delta = 0 

        for state in range(env.nS):
            for action in range(env.nA):
                q_current = Q[state, action]
                q_value = 0

                for prob, next_state, reward, done in env.P[state][action]:
                    q_value += prob * (reward + discount_factor * np.max(Q[next_state]))

                delta = max(delta, abs(q_current - q_value))
                Q[state, action] = q_value

        if delta < theta:
            break

    policy = np.zeros((env.nS, env.nA))
    best_actions = np.argmax(Q, axis=1)  

    for state in range(env.nS):
        policy[state, best_actions[state]] = 1


    return policy, Q
