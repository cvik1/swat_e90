"""
AntAgents.py
This file will contain the definitions of the agents created to solve the
OpenAI Ant problem
"""

import numpy as np
import random

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
        super(DenseAgent, self).__init__(alpha, gamma, epsilon, env)


        self.memory = [] # array to hold experiences for training
        self.net = self.buildNetwork() # call build network to contruct our net

    def getAction(self, state):
        #TODO
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        # adds the most recent experience to memory and formats it
        # accordingly
        raise NotImplementedError


    def trainNet(self):
        # trains the network using our memories
        raise NotImplementedError

    def buildNetwork(self):
        # builds the network
        m = 1 # this is the number of past actions to consider for each action
        input_shape = env.observation_space.shape[0] + m*env.action_space.shape[0]
        output_shape = 1

        # this dictionary contains the layer sizes
        sizes = {
            'l1' = 1200
            'l2' = 1100
            'l3' = 100
        }
        # this dictionary contains the weight variable matrices
        weights = {
            'h1' = tf.Variable(tf.random_normal([input_shape, sizes['l1']]))
            'h2' = tf.Variable(tf.random_normal([sizes['l1', sizes['l2']]]))
            'h3' = tf.Variable(tf.random_normal([sizes['l2'], sizes['l3']]))
            'out' = tf.Variable(tf.random_normal([sizes['l3'], output_shape]))
        }
        # this dictionary containts the bias variable vectors
        bias = {
            'b1' = tf.Variable(tf.random_normal([sizes['l1']]))
            'b2' = tf.Variable(tf.random_normal([sizes['l2']]))
            'b3' = tf.Varialbe(tf.random_normal([sizes['l3']]))
            'out' = tf.Variable(tf.random_normal([output_shape]))
        }
        x = tf.placeholder("float", [None, input_shape])
        y = tf.placeholder("float", [Nonem output_shape])

        # now we actually build out model

        # hidden layer one takes the input
        hidden1 = tf.add(tf.matmul(x, weights['h1']), bias['b1'])
        hidden1 = tf.nn.relu(hidden1) # relu activation

        # hidden layer two takes output from hidden1
        hidden2 = tf.add(tf.matmul(hidden1, weights['h2']), bias['b2'])
        hidden2 = tf.nn.relu(hidden2) # relu activation

        # hidden layer three takes output from hidden2, no activation
        hidden3 = tf.add(tf.matmul(hidden2, weights['h3']), bias['b3'])

        # y (output) takes output from hidden3, no activation
        out = tf.add(tf.matmul(hidden3, weights['out']), bias['out'])

        cost = tf.reduce_mean
