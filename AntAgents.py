"""
AntAgents.py
This file will contain the definitions of the agents created to solve the
OpenAI Ant problem
"""

import numpy as np
import random

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers.merge import Add, Multiply
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

    def trainModel():
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

        self.sess = sess # session to use for training
        self.memory = [] # array to hold experiences for training

        # placeholder to hold the combined gradient of the actor and critic
        # networks for training of the actor network
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])

        # get the actor network and the actor target network
        self.actor_state_input, self.actor = self.buildActor()
        _, self.actor_target = self.buildActor

        # get the actor model gradient for initializing optimizer and training
        actor_weights = self.actor.trainable_weights
        self.actor_grad = tf.gradients(self.actor.output, actor_weights, self.actor_critic_grad)

        # initialize the optimizer for training
        grads = zip(self.actor_grad, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_graidents(grads)

        # get the critic network and the critic target network
        self.critic_state_input, self.critic_action_input, self.critic = self.buildCritic()
        _, _, self.critic_target = self.buildCritic()

        # get the critic model gradient for first round training
        self.critic_grad = tf.gradients(self.critic_model.output, self.critic_action_input)

        # initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def getAction(self, state):
        """
        returns an action from the actor model given the state
        """
        return self.actor.predict(state)

    def buildActor(self):
        """
        defines the actor network
        """
        state_input = Input(shape.self.env.observation_space.shape)
        h1 = Dense(1200, activation='relu')(state_input)
        h2 = Dense(1100, activation='relu')(h1)
        h3 = Dense(100, activation='relu')(h2)
        out = Dense(self.env.action_space.shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=out)
        adam = Adam(lr=self.alpha)
        model.compile(loss='mse', optimizer=adam)

        return state_input, model

    def buildCritic(self):
        """
        defines the critic network
        """
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(1200, activation='relu')(state_input)
        state_h2 = Dense(1100)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1    = Dense(1100)(action_input)

        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(100, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)

        model  = Model(input=[state_input,action_input], output=output)
        adam  = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)

        return state_input, action_input, model

    def trainModel(self):
        """
        trains our models using a batch of experiences from memories
        """
        batch_size = 32
        if len(self.memory < batch_size):
            # if we dont have enough experiences to train
            return

        # sample a batch of experiences randomly from memory
        samples = np.random.choice(self.memory, batch_size)

        # train the networks
        self.trainCritic(samples)
        self.trainActor(samples)
        # after training update the target networks to match actor and critic
        self.updateTarget()

        # decay our epsilon value
        self.epsilon = self.epsilon*.995

    def trainActor(self, samples):
        """
        trains the actor network based on past experiences and the rewards recieved
        and the gradient from training the critic
        since network is in two parts: actor->critic use combined gradient
        """
        for sample in samples:
            state, action, reward, next_state, done = sample
            predicted_action = self.actor.predict(state)

            # get the total combined gradient
            a2c_grad = self.sess.run(self.critic_grad, feed_dict={
                                    self.critic_state_input: state,
                                    self.critic_action_input: predicted_action
                                    })[0]
            # train using the total combined gradient
            self.sess.run(self.optimize, feed_dict{
                                    self.actor_state_input: state,
                                    self.actor_critic_grad: a2c_grad
                                    })

    def trainCritic(self, samples):
        """
        trains the critic network based on the difference between past experiences
        and the outputs of the two target networks
        """
        for sample in samples:
            state, action, reward, next_state, done = sample
            if not done:
                # get the predicted action from the state
                target_action = self.actor_target.predict(next_state)
                # get the predicted next value of the state action pair
                target_reward = self.critic_target.predict([next_state,target_action][0][0])
                # dicount reward using discount rate
                reward = self.gamma*target_reward
            # fit the critic network to the experience and estimated future reward
            self.critic.fit([state, action], reward, verbose=0)

    def updateTarget(self):
        """
        update the target networks after a training round has been completed
        """
        # update the critic target weights
        critic_weights  = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()

        for i in range(len(critic_target_weights)):
        	critic_target_weights[i] = critic_weights[i]
        self.critic_target.set_weights(critic_target_weights)

        # update the actor weights
        actor_weights  = self.actor.get_weights()
        actor_target_weights = self.target_critic.get_weights()

        for i in range(len(actor_target_weights)):
        	actor_target_weights[i] = actor_weights[i]
        self.target_critic.set_weights(actor_target_weights)
