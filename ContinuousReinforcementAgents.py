"""
ContinuousReinforcementAgents.py
This file will define the class ContinuousQLearningAgent as:
An agent that approximates states to maintain a table of (state,action) pairs
in order to estimate Q-values from experience rather than a model
"""

import numpy as np
import gym
import pickle
import random
import math
import itertools

import ReinforcementAgents

class ContinuousQLearningAgent(ReinforcementAgents.QLearningAgent):

    def __init__(self, alpha, gamma, env, buckets, a_buckets):
        """
        Initializes our basic ContinuousQLearningAgent
        """
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor

        self.Q_values = {} # dictionary to hold Q values
        self.env = env #learning environment
        self.buckets = buckets # number of buckets for discretizing state space
        self.a_buckets = a_buckets # number of buckets for discretizing actions
        self.disc_actions = self.getDiscreteActions(5) # array to hold discretized actions

        high = env.observation_space.low
        low = env.observation_space.high
        self.state_buckets = [0 for i in range(len(low))]
        # make the list of buckets to approximate our states
        for i in range(len(low)):
            try:
                self.state_buckets[i] = np.arange(low[i],high[i],(high[i]-low[i])/buckets)
            # if range is too large to compute buckets
            except:
                self.state_buckets[i] = np.arange(-10, 10, 20/buckets)

    def setQValues(self, filename):
        """
        if loading a pretrained agent, set the Q-value table to the
        table from that agent's training
        """
        try:
            f = open(filename, 'rb')
            # load the dictionary form the file
            self.Q_values = pkl.load(f)
            return True
        except:
            return False

    def saveQValues(self, filename):
        """
        save Q values to a file for reloading at another time
        """
        try:
            f = open(filename, 'wb')
            # dump the dictionary into the file
            pickle.dump(self.Q_values, f)
            return (True, None)
        except Exception as e:
            return (False, e)

    def getQValues(self):
        """
        returns Q values for debugging purposes
        """
        return self.Q_values

    def getDiscreteActions(self, buckets):
        """
        Gets a list of discretized actions for our agent to choose from
        """
        # set the number of discretization buckets
        # buckets = 4
        # get the action space range
        try:
            low = self.env.action_space.low
            high = self.env.action_space.high
            # initialize the arrays for actions
            action_buckets = [0 for i in range(len(low))]
            # create the arrays of buckets for the actions
            for i in range(len(low)):
                try:
                    action_buckets[i] = tuple(np.arange(low[i],high[i],(high[i]-low[i])/buckets))
                # if range is too large to compute buckets
                except:
                    action_buckets[i] = tuple(np.arange(-2, 2, 4/buckets))
            if len(action_buckets) == 1:
                disc_actions = action_buckets[0]
            elif len(action_buckets) == 2:
                # get all possible combinations of actions
                disc_actions = list(itertools.product(action_buckets[0],
                                                        action_buckets[1]))
            # return the discretized actions
            return disc_actions
        except:
            return None

    def greedyPolicy(self, state):
        """
        Chooses the best availble action based on Q-values
        inputs: state
        outputs: action
        """
        try:
            actions = range(0, self.env.action_space.n)
        except:
            actions = self.disc_actions
        if len(actions) == 0:
            return None
        values = {}
        # gets the expected value for each action
        for action in actions:
            values[self.actionValue(state, action)] = action
        keys = list(values.keys())
        random.shuffle(keys) # shuffle so in case of a tie we choose randomly
        return values[max(keys)]

    def getAction(self, state):
        """
        the method called by the simulation script to get an action
        input: state
        output: action from greedy policy
        """
        # get the closest state approximation for the q-table
        state_approx = state
        for i in range(len(state)):
            state_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-state[i]))

        action = self.greedyPolicy(tuple(state_approx))

        return action

    def explore(self, state):
        """
        the method called by the simulation script to get an action during exploration
        input: state
        output: action from exploration policy
        """
        # get the closest state approximation for the q-table
        state_approx = state
        for i in range(len(state)):
            state_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-state[i]))

        action = self.explorationPolicy(tuple(state_approx))

        return action

    def update(self, state, action, nextState, reward, done):
        """
        update the Q table using the reward
        """
        # get the approximate states for the q-table updates
        state_approx = state
        nextState_approx = nextState
        for i in range(len(state)):
            state_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-state[i]))
            nextState_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-nextState[i]))
        # convert to tuples for the dictionary
        state_approx = tuple(state_approx)
        nextState_approx = tuple(nextState_approx)

        # update the q table
        nextQ = self.stateValue(nextState_approx)
        curQ = self.actionValue(state_approx, action)
        self.Q_values[(state_approx, action)] = (self.actionValue(state_approx, action) +
                self.alpha * (reward + self.gamma * nextQ - curQ))

class GreedyAgent(ContinuousQLearningAgent):

    def __init__(self, alpha, gamma, epsilon, env, buckets):
        """
        Initializes our Epsilon greedy q-learning agent
        """
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor
        self.epsilon = float(epsilon) #exploration randomization factor

        self.Q_values = {}
        self.env = env #learning environment
        self.disc_actions = self.getDiscreteActions(5) # array to hold discretized actions

        high = list(env.observation_space.low)
        low = list(env.observation_space.high)
        self.state_buckets = [0 for i in range(len(low))]
        # make the list of buckets to approximate our states
        for i in range(len(low)):
            try:
                self.state_buckets[i] = np.arange(low[i],high[i],(high[i]-low[i])/buckets)
            # if range is too large to compute buckets
            except:
                self.state_buckets[i] = np.arange(-10, 10, 20/buckets)


    def explorationPolicy(self, state):
        """
        implements the epsilon greedy exploration policy
        inputs: state
        outputs: action
        """
        try:
            actions = range(0, self.env.action_space.n)
        except:
            actions = self.disc_actions
        if len(actions) == 0:
            return None
        values = {}
        # gets the expected value for each action
        for action in actions:
            values[self.actionValue(state, action)] = action
        keys = list(values.keys())
        random.shuffle(keys) # shuffle so in case of a tie we choose randomly
        ran = random.random()
        if ran < self.epsilon: # if random value less than epsilon
            return random.choice(list(values.values())) # choose a random action
        else: # otherwise choose the greedy action
            return values[max(keys)]

    def update(self, state, action, nextState, reward, done):
        """
        update the Q table using the reward
        """
        # get the approximate states for the q-table updates
        state_approx = state
        nextState_approx = nextState
        for i in range(len(state)):
            state_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-state[i]))
            nextState_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-nextState[i]))
        # convert to tuples for the dictionary
        state_approx = tuple(state_approx)
        nextState_approx = tuple(nextState_approx)

        # if we're still learning update the q table
        nextQ = self.stateValue(nextState_approx)
        curQ = self.actionValue(state_approx, action)
        self.Q_values[(state_approx, action)] = (self.actionValue(state_approx, action) +
                self.alpha * (reward + self.gamma * nextQ - curQ))

class UCBAgent(ContinuousQLearningAgent):

    def __init__(self, alpha, gamma, UCB_const, env, buckets):
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.UCB_const = float(UCB_const) # constant to calculate UCB values in exploration

        self.Q_values = {}
        self.visits = {}
        self.env = env
        self.disc_actions = self.getDiscreteActions(5) # array to hold discretized actions

        high = env.observation_space.low
        low = env.observation_space.high
        self.state_buckets = [0 for i in range(len(low))]
        # make the list of buckets to approximate our states
        for i in range(len(low)):
            try:
                self.state_buckets[i] = np.arange(low[i],high[i],(high[i]-low[i])/buckets)
            # if range is too large to compute buckets
            except:
                self.state_buckets[i] = np.arange(-10, 10, 20/buckets)

    def explorationPolicy(self, state):
        """
        Implements UCB exploration policy
        Computes UCB weights, then normalizes them into a probability
        distribution to sample from
        inputs: state
        outputs: action
        """

        try:
            actions = range(0, self.env.action_space.n)
        except:
            actions = self.disc_actions
        if len(actions) == 0:
            return None
        weights = []

        action_visits = []
        # count how many times each action has been taken from this state
        for action in actions:
            w = self.actionValue(state, action)
            v = self.visits.get((state,action), 0)
            weights.append(w)
            action_visits.append(v)
        # sum of all actions taken from this state equals total visits to the state
        sum_v = sum(action_visits)
        if sum_v != 0:
            # if not the first visit to state, compute UCB weights
            for i in range(len(action_visits)):
                if action_visits[i] != 0:
                    ucb = self.UCB_const * math.sqrt(math.log(sum_v) / action_visits[i])
                else:
                    ucb = self.UCB_const * math.sqrt(math.log(sum_v) / 1)
                weights[i] += ucb

        sum_w = float(sum([abs(w) for w in weights]))
        if sum_w == 0:
            # if the first visit to this state choose action at random
            index = np.random.choice(range(len(weights)))
        else: # create distribution using UCB weights and sample
            # normalize each weight by the sum
            norm_weights = [abs(i)/sum_w for i in weights]
            # randomly select the index of a weight from the normalized weights
            index = np.random.choice(range(len(weights)),1, p=norm_weights)[0]

        return actions[index]

    def update(self, state, action, nextState, reward, done):
        """
        update the Q table using the reward
        update the visits table
        """
        # get the approximate states for the q-table updates
        state_approx = state
        nextState_approx = nextState
        for i in range(len(state)):
            state_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-state[i]))
            nextState_approx[i] = min(self.state_buckets[i], key=lambda x:abs(x-nextState[i]))
        # convert to tuples for the dictionary
        state_approx = tuple(state_approx)
        nextState_approx = tuple(nextState_approx)

        # update the q table
        nextQ = self.stateValue(nextState_approx)
        curQ = self.actionValue(state_approx, action)
        self.Q_values[(state_approx, action)] = (self.actionValue(state_approx, action) +
                    self.alpha * (reward + self.gamma * nextQ - curQ))

        self.visits[(state_approx,action)] = self.visits.get((state_approx,action),0) + 1
