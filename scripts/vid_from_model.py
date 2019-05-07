"""
vid_from_model.py
this script will save videos of runs from loaded model weights
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

import AntAgents

def main():
    print(sys.argv)


    alpha = .003
    gamma = .9
    epsilon = .9995
    env = gym.make("RoboschoolAnt-v1")


    agent = AntAgents.A2CAgent_v2_tf(alpha, gamma, epsilon, env)

    agent.saver.restore(agent.sess, "model_r661.2686468345071.data-00000-of-00001.ckpt")

    monitor = gym.wrappers.Monitor(env, "videos/vid_"+str(file), force=True)

    state = monitor.reset()
    state = state.reshape(1,-1)
    action = None
    done = False
    while not done:
        # get an action from the exploration policy
        action = agent.explore(state)

        # apply the action to the env
        # we must reshape the action before stepping for compatability with
        # rendering/recording the video
        next_state, reward, done, info = monitor.step(action.reshape(1,-1)[0])
        # update state for next iteration
        state = next_state.reshape(1,-1)

    monitor.close()



if __name__ == "__main__" :
    main()
