"""
train.py
this file will train agents to solve the RoboschoolAnt-v1 problem
using deep reinforcement learning methods
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

def main():
    """
    train a model
    """
    # make the argument parser
    parser  = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", help="which agent version to use", type=int)
    parser.add_argument("-s", "--save", action='store_true', help="save model after training")
    parser.add_argument("-l", "--load", help="load a model to use")
    parser.add_argument("-n", "--numEpisodes", help="number of episodes to train (default 1000)",
                        type=int)
    parser.add_argument("-a", "--actor", help="file where actor model is saved")
    parser.add_argument("-c", "--critic", help="file where critic model is saved")
    parser.add_argument("-f", "--savefile", help="base path where to save models")
    parser.add_argument("-r", "--render", action='store_true', help="flag whether or not to render video")
    # parse the arguments

    args = parser.parse_args()
    # define variables to initialize the agent
    alpha = .003
    gamma = .99
    epsilon = .9995

    # env = gym.make('AntPyBulletEnv-v0')
    # env = gym.make("RoboschoolAnt-v1")
    # env = gym.make("Pendulum-v0")
    env = gym.make("MountainCarContinuous-v0")

    agent = AntAgents.A2CAgent_v2_tf(alpha, gamma, epsilon, env)
    # agent = AntAgents.DenseAgent_v2(alpha, gamma, epsilon, env)

    print("Agent Initialized")

    # agent.actor.save_weights("init_actor.hd5")
    # agent.critic.save_weights("init_critic.hd5")

    # number of episodes to train on
    if args.numEpisodes == None:
        training_episodes = 1000
    else:
        training_episodes = args.numEpisodes

    file = open("output.out", 'w')
    data = np.empty((training_episodes,2))

    print("Beginning training...\n")
    sys.stdout.flush()

    # if we don't get a positive reward from training within the first n episodes
    # we have likely fallen into a local minimum and need to restart

    training = True
    # to tell us whether or not we have gotten a positive reward
    pos_reward = False

    # to tell us when we're reached a new max reward
    max_reward = 0

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
                # while not done:
                while not done:
                    # get an action from the exploration policy
                    action = agent.explore(state)

                    # apply the action to the env
                    # we must reshape the action before stepping for compatability with
                    # rendering/recording the video
                    next_state, reward, done, info = env.step(action.reshape(1,-1)[0])

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

                    # online training
                    agent.online(state, action, reward, next_state, done)

                # train the model after every iteration
                # agent.trainModel()

                # write result to data
                data[episode,:] = [sum_reward, steps]
                # if (steps > (env._max_episode_steps/2)) :
                #     print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                #             episode+1, sum_reward, steps))

                if (reward > .6*max_reward) :
                    print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))

                # print the statistics from the training episode
                if (episode+1)%(training_episodes//100) == 0:
                    print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))
                    # file.write("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps\n".format(
                    #         episode+1, sum_reward, steps))
                    sys.stdout.flush()
                    # print("Epsilon={:1.4f}".format(agent.epsilon))
                # if we've gotten positive reward record it
                if sum_reward > 0 and pos_reward == False:
                    # print("Results from episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                    #         episode+1, sum_reward, steps))
                    print("\n WE HAVE CLEARED THE HURDLE!!\n")
                    pos_reward = True

                # if we havent gotten a positive reward in the first n episodes
                # restart
                if not pos_reward and episode>(20):
                    print("\nRestarting Training...")
                    sys.stdout.flush()
                    # set the weights to initial values and restart training
                    agent.sess.run(tf.global_variables_initializer())
                    break

                # if we have gotten a new best run by > 20%, save the weights from that run
                if sum_reward>(1.1*max_reward):
                    max_reward = sum_reward
                    # agent.saver.save(agent.sess, "models/model_r"+str(sum_reward)+".ckpt")
                    # # now save the array
                    # np.savetxt('test_results/test_'+str(episode)+'_'+str(sum_reward)+'.csv', test_record, delimiter=',')

                # # rather than rendering we will record the actions taken so we can
                # # render later on
                # if (episode+1)%(training_episodes//100) == 0:
                #     test_record = np.empty((1000,36))
                #     # initialize environment variables
                #     state = env.reset()
                #     # reshape the state array
                #     state = state.reshape(1,-1)
                #     sum_reward = 0
                #     steps = 0
                #     done = False
                #     action = None
                #     # while not done:
                #     while not done:
                #         # get an action from the exploration policy
                #         action = agent.explore(state)
                #
                #         # apply the action to the env
                #         # we must reshape the action before stepping for compatability with
                #         # rendering/recording the video
                #         next_state, reward, done, info = env.step(action.reshape(1,-1)[0])
                #
                #         # place the action in the array
                #         test_record[steps,:8] = action.reshape(1,-1)[0].flatten()
                #         # place the state in the array
                #         test_record[steps,8:] = state.flatten()
                #
                #         # reshape the action for input use in training
                #         action = action.reshape(1, -1)
                #         # reshape the next_state array
                #         next_state = next_state.reshape(1,-1)
                #         # add this experience to memory for training use later
                #         agent.remember(state, action, reward, next_state, done)
                #         # update the current state
                #         state = next_state
                #         # update the steps and sum reward for bookkeeping purposes
                #         steps +=1
                #         sum_reward += reward
                #     # now save the array
                #     np.savetxt('test_results/test_'+str(episode)+'_'+str(sum_reward)+'.csv', test_record, delimiter=',')
                #
                # after every 100 episodes, we want to render a test episode to save
                # if the render flag is high
                if args.render and (episode+1)%(training_episodes//10) == 0:
                    # monitor object will allow us to save the video
                    # create directory "training_2100" where 2100 is the episode number
                    monitor = gym.wrappers.Monitor(env, "videos/car/training_"+str(episode+1), force=True)
                    # initialize environment variables
                    state = monitor.reset()
                    # reshape the state array
                    state = state.reshape(1,-1)
                    sum_reward = 0
                    steps = 0
                    done = False
                    action = None
                    while not done:
                        # get an action from the greedy policy
                        action = agent.explore(state)
                        # apply the action to the env
                        # we must reshape action for rendering purposes
                        next_state, reward, done, info = monitor.step(action.reshape(1,-1)[0])

                        # reshape the next_state array
                        next_state = next_state.reshape(1,-1)
                        # update state
                        state = next_state

                        # update the steps and sum reward for bookkeeping purposes
                        steps +=1
                        sum_reward += reward

                    monitor.close()


                    # print results from test episode
                    print("Results from test episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))
                    file.write("Results from test episode {:6d}: Total Reward={:7.2f} over {:3d} steps".format(
                            episode+1, sum_reward, steps))


            if pos_reward:
                break
    t2 = datetime.now()
    print((t2-t1).seconds)

    np.savetxt('data.csv', data, delimiter=',')


    # print("testing model and rendering test...")
    #
    # # create a new monitor
    # monitor = gym.wrappers.Monitor(env, "testing", force=True)
    # # after training run a testing episode to see how we do
    # state = monitor.reset()
    # # reshape the state array for passing into our model
    # state = state.reshape((1,env.observation_space.shape[0]))
    #
    # sum_reward = 0
    # steps = 0
    # done = False
    # action = None
    # while not done:
    #     # get an action from the greedy policy
    #     action = agent.getAction(state)
    #     # apply the action to the env
    #     # we must reshape action for rendering purposes
    #     next_state, reward, done, info = monitor.step(action.reshape(1,-1)[0])
    #
    #     # # render so we can see the strategy learned by the agent
    #     # monitor.render()
    #
    #     # reshape the next_state array
    #     next_state = next_state.reshape((1,env.observation_space.shape[0]))
    #     # update state
    #     state = next_state
    #
    #     # update the steps and sum reward for bookkeeping purposes
    #     steps +=1
    #     sum_reward += reward
    #
    # print("Results from test episode: Total Reward={:7.2f} over {:3d} steps".format(
    #         sum_reward, steps))
    # file.write("Results from test episode: Total Reward={:7.2f} over {:3d} steps".format(
    #         sum_reward, steps))
    #
    # env.close()
    # monitor.close()

    # print(args.save)
    # # if we are saving weights
    # if args.save:
    #     # if we were given a path to use to save th weights
    #     if args.savefile != None:
    #         actor_path = args.savefile
    #         critic_path = args.savefile
    #     else:
    #         actor_path = ""
    #         critic_path = ""
    #     actor_path += "actor.hd5"
    #     critic_path += "critic.hd5"
    #
    #     agent.saveWeights(actor_path, critic_path)



if __name__=="__main__":
    main()
