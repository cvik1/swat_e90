"""
DeepReinforcementAgents.py
This file will define the class DeepReinforcementAgent as:
"""

import numpy as np
import random
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

class DeepReinforcementAgent():
    """
    Defines the general class for our DeepReinforcmentAgent
    Only the explore method will be filled out
    All other methods will be overwritten in the extended classes defined below
    """

    def __init__(self, alpha, gamma, epsilon, env):
        """
        Initializes our basic deep learning agent
        """
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.memory = [] # array to store experiences for training
        self.epsilon = float(epsilon)
        self.env = env # environment to train on

        self.model = self.buildNetwork() # create our nextwork
        # self.session = tf.Session() # define the session
        # self.session.run(tf.global_variables_initializer()) # initialize our model


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

    def update(self, state, action, reward, next_state, done):
        """
        wrapper for self.remember() to match other classes of agents
        """
        self.remember(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        """
        Adds training experiences to our memory
        """
        # make sure the states are the correct shape for training
        state = tf.reshape(state, [1,-1])
        next_state = tf.reshape(next_state, [1,-1])
        # append the experience to memory
        self.memory.append((state,action,reward,next_state,done))

    def save_weights(self, filename):
        """
        Saves our model weights to file to be loaded later
        """
        # TODO

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


    def buildNetwork(self):
        """
        Defines the architecture of our neural network
        """
        # to be overwritten
        raise Exception('Function not implemented')




class TaxiAgent(DeepReinforcementAgent):
    """
    Agent that uses a neural network to learn the Q function for Taxi-v2
    """
    def __init__(self, alpha, gamma, epsilon, env):
        """
        Initializes our agent
        """
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.epsilon = float(epsilon)
        self.memory = [] # array to store experiences for training
        self.env = env # environment to train on

        self.model = self.buildNetwork() # create our nextwork
        # self.session = tf.Session() # define the session
        # self.session.run(tf.global_variables_initializer()) # initialize our model

    def buildNetwork(self):
        """
        Defines the architecture of our neural network
        """
        input_size = [1, self.env.observation_space.n]
        action_size = [1, self.env.action_space.n]

        model = Sequential()
        model.add(Dense(24, input_dim=500, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model

    def getAction(self, state):
        """
        returns an action based on the prediction of our nextwork
        taxi is a discrete environment so this function is modified
        """
        # create one-hot array to represent state
        input = np.zeros([1,self.env.observation_space.n])
        input[0,state] = 1
        q_values = self.model.predict(input, steps=1)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Adds training experiences to our memory
        """
        # create one-hot arrays to store the states
        s = np.zeros([1,self.env.observation_space.n])
        s[0,state] = 1
        ns = np.zeros([1,self.env.observation_space.n])
        ns[0,next_state] = 1
        # append the experience to memory
        self.memory.append((s,action,reward,ns,done))

class CartPoleAgent(DeepReinforcementAgent):
    """
    Agent that uses a neural network to learn the Q function for CartPole-v1
    """
    def __init__(self, alpha, gamma, epsilon, env):
        """
        Initializes our agent
        """
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.epsilon = float(epsilon) # chance of exploring with random action
        self.memory = [] # array to store experiences for training
        self.env = env # environment to train on

        self.model = self.buildNetwork() # create our nextwork
        # self.session = tf.Session() # define the session
        # self.session.run(tf.global_variables_initializer()) # initialize our model

    def buildNetwork(self):
        """
        Defines the architecture of our neural network
        """
        input_size = [1,len(self.env.observation_space.low)]
        action_size = [1, self.env.action_space.n]

        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model

class MountainCarAgent(DeepReinforcementAgent):
    """
    Agent that uses a neural network to learn the Q function for MountainCar-v0
    """
    def __init(self, alpha, gamma, epsilon, env):
        """
        Initializes our agent
        """
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.epsilon = float(epsilon) # chance of exploring with random action
        self.memory = [] # array to store experiences for training
        self.env = env # environment to train on

        self.model = self.buildNetwork() # create our nextwork
        # self.session = tf.Session() # define the session
        # self.session.run(tf.global_variables_initializer()) # initialize our model

    def buildNetwork(self):
        """
        Defines the network architecture of our neural network
        """
        input_size = [1,len(self.env.observation_space.low)]
        action_size = [1, self.env.action_space.n]

        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model

class LunarLanderAgent(DeepReinforcementAgent):
    """
    Agent that uses a neural network to learn the Q function for LunarLander-v0
    """
    def __init(self, alpha, gamma, epsilon, env):
        """
        Initializes our agent
        """
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.epsilon = float(epsilon) # chance of exploring with random action
        self.memory = [] # array to store experiences for training
        self.env = env # environment to train on

        self.model = self.buildNetwork() # create our nextwork
        # self.session = tf.Session() # define the session
        # self.session.run(tf.global_variables_initializer()) # initialize our model

    def buildNetwork(self):
        """
        Defines the architecture of our neural network
        """
        input_size = [1,len(self.env.observation_space.low)]
        action_size = [1, self.env.action_space.n]

        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model
