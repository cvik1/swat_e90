"""
AntAgents.py
This file will contain the definitions of the agents created to solve the
OpenAI Ant problem
"""

import numpy as np
import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

class MLAgent():
    """
    This defines the generic class of a machine learning agent for the purposes
    of this project
    Each of the subsequent agents will extend or overwrite the generic functions
    defined here
    """
    def __init__(self, alpha, gamma, epsilon, env):
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.memory = [] # array to store experiences for training
        self.epsilon = float(epsilon)
        self.env = env # environment to train on


    def getAction(self, state):
        # returns an action based on the policy and the current state
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        # gets observation from last action
        raise NotImplementedError

    def buildNetwork(self):
        # builds the networks for our agent if needed
        raise NotImplementedError

    def saveNet(self):
        # saves the network architecture and weights
        raise NotImplementedError

    def loadNet(self):
        # loads the network architecture and weights from file
        raise NotImplementedError

    def saveMemory(self):
        # loads experiences saved to a file for training
        raise NotImplementedError


class RandomAgent(MLAgent):
    """
    This defines an agent that will always choose random actions
    """

    def __init__(self, env):
        self.env = env

    def getAction(self, state):
        # return random action sampled from action space
        return self.env.action_space.sample()

class DenseAgent(MLAgent):
    """
    This defines an agent that will use a simple dense network architecture to
    learn
    """
    def __init__(self, alpha, gamma, epsilon, env):
        # use the base class definition to initialize
        super().__init__(alpha, gamma, epsilon, env)


        self.memory = [] # array to hold experiences for training
        self.net = self.buildNetwork() # call build network to contruct our net

    def explore(self, state):
        """
        returns an action based on the epsilon greedy exploration policy
        """
        if np.random.rand() < self.epsilon:
            # get random action for exploration
            action = self.env.action_space.sample()
        else:
            # get action based on q values from network
            action = self.getAction(state)
        return action

    def getAction(self,state):
        """
        returns an action based on the result from the neural network
        """
        input = tf.reshape(state, [1,-1])
        q_values = self.model.predict(input, steps=1)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Adds training experiences to our memory
        """
        # make sure the states are the correct shape for training
        state = tf.reshape(state, [1,-1])
        next_state = tf.reshape(next_state, [1,-1])
        # append the experience to memory
        self.memory.append((state,action,reward,next_state,done))

    def train_model():
        """
        Trains our model based on experiences
        """
        x_batch = [] # array to hold our input batch
        y_batch = [] # array to hold our ouput batch
        # sample from memory to get experiences to use for training
        if len(self.memory) > 64:
            batch = random.sample(self.memory, 64)
        else:
            batch = self.memory
        for state, action, reward, next_state, done in batch:
            # create our target with appropriate reward and prediction
            target = reward
            if not done:
                target = self.gamma * self.getAction(state)
            y_target = self.model.predict(state)
            y_target_f[0][action] = target

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        # now train model on x_batch and y_batch
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=(len(x_batch)), verbose=0)

        # decay our value for epsilon
        self.epsilon = self.epsilon * .995
        # decau our value for gamma
        self.gamma = self.gamma * .995

    def buildNetwork(self):
        """
        defines the network
        """
        input_shape = env.observation_space.shape[0]
        # the number of possible actions
        output_shape = 1

        input_size = [1,len(self.env.observation_space.low)]
        action_size = [1, self.env.action_space.n]

        model = Sequential()
        model.add(Dense(1200, input_dim=input_shape, activation='relu'))
        model.add(Dense(1100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model


class A2CAgent(MLAgent):
    """
    This agent will implement an asynchronous actor-critic network architecture
    to solve the OpenAI Ant problem
    """
    def __init__(self, alpha, gamma, epsilon, env):
        super().__init__(alpha, gamma, epsilon, env)

        self.actor_alpha = self.alpha
        self.critic_alpha = self.alpha
        self.memory = []
        # get the networks that describe the actor: mean and variance of actions
        self.mu, self.var= self.buildActor()
        # get the network that describes the critic: value of states
        self.critic = self.buildCritic()

    def buildActor(self):
        """
        defines the actor network
        """

        input_shape = [1,len(self.env.observation_space.low)]
        output_shape = [1, len(self.env.action_space.low)]

        base = Sequential()
        base.add(Dense(128, input_dim=input_shape, activation='relu'))

        mu = Sequential()
        mu.add(Dense(output_shape, activation='linear'))
        mu.compile(loss='mse',
                      optimizer=Adam(lr=self.actor_alpha))

        var = Sequential()
        var.add(Dense(output_shape, activation='linear'))
        var.compile(loss='mse',
                    optimizer=Adam(lr=self.actor_alpha))

        return mu, var


    def buildCritic(self):
        """
        defines the critic network
        """
        input_shape = [1,len(self.env.observation_space.low)]
        output_shape = [1, len(self.env.action_space.low)]

        critic = Sequential()
        critic.add(Dense)
