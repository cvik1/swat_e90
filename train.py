"""
train.py
this file will train agents to solve the RoboschoolAnt-v1 problem
using deep reinforcement learning methods
"""
# import training envionments
import gym
import roboschool
# import agents
import AntAgents

import sys
import argparse
import numpy as np

def main():
    """
    train a model
    """
    # make the argument parser
    parser  = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", action='store_true', help="save model after training")
    parser.add_argument("-l", "--load", help="load a model to use")
    parser.add_argument("-n", "--numEpisodes", help="number of episodes to train (default 1000)",
                        type=int)
    parser.add_argument("-a", "--actor", help="file where actor model is saved")
    parser.add_argument("-c", "--critic", help="file where critic model is saved")
    parser.add_argument("-f", "--savefile", help="base path where to save models")
    # parse the arguments

    args = parser.parse_args()
    # define variables to initialize the agent
    alpha = .001
    gamma = .9
    epsilon = .8
    env = gym.make("RoboschoolAnt-v1")

    # now create the agent
    agent = AntAgents.A2CAgent(alpha, gamma, epsilon, env)

    if args.load == None:
        # if we are not loading a model train a new one

        # number of episodes to train on
        if args.numEpisodes == None:
            training_episodes = 1000
        else:
            training_episodes = args.numEpisodes

        print("Beginning training...\n")

        # train the agent
        for episode in range(training_episodes):
            # initialize environment variables
            state = env.reset()
            # reshape the state array
            state = state.reshape((1,env.observation_space.shape[0]))
            sum_reward = 0
            steps = 0
            done = False
            action = None
            while not done:
                # get an action from the exploration policy
                action = agent.explore(state)

                # apply the action to the env
                # we must reshape the action before stepping for compatability with
                # rendering/recording the video
                next_state, reward, done, info = env.step(action.reshape(1,-1)[0])

                # reshape the action for input use in training
                action = action.reshape((1, env.action_space.shape[0]))
                # reshape the next_state array
                next_state = next_state.reshape((1,env.observation_space.shape[0]))
                # add this experience to memory for training use later
                agent.remember(state, action, reward, next_state, done)
                # update the current state
                state = next_state
                # update the steps and sum reward for bookkeeping purposes
                steps +=1
                sum_reward += reward

            # print the statistics from the training episode
            if (episode+1)%(training_episodes//10) == 0:
                print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                        episode+1, sum_reward, steps))
            # train the model after every iteration
            agent.trainModel()

    else:
        print("loading model...")

        agent.loadWeights(args.load)


    print("testing model and rendering test...")

    # after training run a testing episode to see how we do
    state = env.reset()
    # reshape the state array for passing into our model
    state = state.reshape((1,env.observation_space.shape[0]))

    sum_reward = 0
    steps = 0
    done = False
    action = None
    while not done:
        # get an action from the greedy policy
        action = agent.getAction(state)
        # apply the action to the env
        # we must reshape action for rendering purposes
        next_state, reward, done, info = env.step(action.reshape(1,-1)[0])

        # render so we can see the strategy learned by the agent
        env.render()

        # reshape the next_state array
        next_state = next_state.reshape((1,env.observation_space.shape[0]))
        # update state
        state = next_state

        # update the steps and sum reward for bookkeeping purposes
        steps +=1
        sum_reward += reward

    print("Results from test episode: Total Reward={:7.2f} over {:3d} steps".format(
            sum_reward, steps))

    # if we are saving weights
    if args.save != None:
        # if we were given a path to use to save th weights
        if args.savefile != None:
            actor_path = args.savefile
            critic_path = args.savefile
        else:
            actor_path = ""
            critic_path = ""
        actor_path += "actor.hd5"
        critic_path += "critic.hd5"

        agent.saveWeights(actor_path, critic_path)



if __name__=="__main__":
    main()
