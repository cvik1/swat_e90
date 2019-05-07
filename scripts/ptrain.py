"""
ptrain.py
trains policy gradient
"""
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# import training envionments
import gym
# import monitor for saving video
from gym.wrappers import Monitor
import roboschool
# import pybulletgym


# import agents
import AntAgents

def discount_reward(r):
    """
    gets the total discounted reward over the episode
    """
    discounted_r = np.zeros_like(r)
    sum = 0
    for i in reversed(range(len(r))):
        sum = sum * .99 + r[i]
        discounted_r[i] = sum
    m = np.mean(discounted_r)
    std = np.std(discounted_r)
    normal_r = (discounted_r-m)/std
    return discounted_r
    # return normal_r

def main():
    """
    trains our agent
    """

    # make the argument parser
    parser  = argparse.ArgumentParser()
    parser.add_argument("-n", "--numEpisodes", help="number of episodes to train (default 1000)",
                        type=int)
    args = parser.parse_args()


    # number of episodes to train on
    if args.numEpisodes == None:
        training_episodes = 1000
    else:
        training_episodes = args.numEpisodes


    alpha = .003
    gamma = .99
    epsilon = .9995

    # env = gym.make('AntPyBulletEnv-v0')
    # env = gym.make("RoboschoolAnt-v1")
    # env = gym.make("Pendulum-v0")
    env = gym.make("MountainCarContinuous-v0")

    agent = AntAgents.PolicyGradientAgent(alpha, gamma, epsilon, env)

    data = np.empty((training_episodes,2))

    print("Beginning training...\n")
    sys.stdout.flush()

    # if we don't get a positive reward from training within the first n episodes
    # we have likely fallen into a local minimum and need to restart

    training = True
    # to tell us whether or not we have gotten a positive reward
    pos_reward = False

    # to tell us when we're reached a new max reward
    max_reward = -np.inf
    # array to hold data for multiple episodes for use in testing
    e = []

    t1 = datetime.now()
    while training:

            # train the agent
            for episode in range(training_episodes):
                # print("Episode: {}".format(episode))
                # initialize environment variables
                state = env.reset()
                # reshape the state array
                state = state.reshape(1,-1)
                sum_reward = 0
                steps = 0
                done = False
                action = None
                # to record the episode
                test_record = np.empty((1000,36))
                # to store rewards, states, actions at each step
                r = []
                s = []
                a = []
                # while not done:
                while not done:
                    # get an action from the exploration policy
                    action = agent.explore(state)

                    # apply the action to the env
                    # we must reshape the action before stepping for compatability with
                    # rendering/recording the video
                    next_state, reward, done, info = env.step(action.reshape(1,-1)[0])

                    s.append(state)
                    a.append(action)
                    r.append(reward)

                    # reshape the action for input use in training
                    action = action.reshape(1, -1)
                    # reshape the next_state array
                    next_state = next_state.reshape(1,-1)
                    # place the action in the array
                    # test_record[steps,:8] = action.reshape(1,-1)[0].flatten()
                    # # place the state in the array
                    # test_record[steps,8:] = state.flatten()
                    # add this experience to memory for training use later
                    agent.remember(state, action, reward, next_state, done)
                    # update the current state
                    state = next_state
                    # update the steps and sum reward for bookkeeping purposes
                    steps +=1
                    sum_reward += reward

                # get discounted rewards
                r_d = discount_reward(r)

                # print("\nSHAPES:")
                # print(np.array(s).shape, np.array(a).shape, np.array(r).shape)

                # add this episode to our memory array
                # e.append([s,a,r])

                # train the model every fifth iteration
                if episode%5 ==0:
                    # train
                    agent.trainModel(s, a, r_d)

                # write result to data
                data[episode,:] = [sum_reward, steps]

                # if we have gotten a new best reward
                if reward>(1.1*max_reward):
                    max_reward = reward
                    # print stats
                    print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))
                    sys.stdout.flush()
                    # save the episode results
                    np.savetxt('test_results/test_'+str(episode)+'_'+str(sum_reward)+'.csv', test_record, delimiter=',')

                if (reward >= .5*max_reward) :
                    print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))
                    sys.stdout.flush()

                # print the statistics from the training episode
                if (episode+1)%(training_episodes//100) == 0:
                    print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))
                    sys.stdout.flush()

                if sum_reward > 50 and pos_reward == False:
                    # print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                    #         episode+1, sum_reward, steps))
                    print("\n WE HAVE CLEARED THE HURDLE!!\n")
                    pos_reward = True
                if not pos_reward and episode>(training_episodes//5):
                    print("\nRestarting Training...")
                    sys.stdout.flush()
                    # set the weights to initial values and restart training
                    agent.sess.run(tf.global_variables_initializer())
                    break

            if pos_reward:
                break

    t2 = datetime.now()
    print((t2-t1).seconds)

    np.savetxt('data.csv', data, delimiter=',')


if __name__ == "__main__":
    main()
