"""
make_videos.py
this script will produce videos from data files and/or
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

    env = gym.make("RoboschoolAnt-v1")

    for file in sys.argv[1:]:
        env.reset()

        data = np.loadtxt(file, dtype=np.float32, delimiter=",")

        actions = data[:,:8]
        states = data[:,8:]

    
        monitor = gym.wrappers.Monitor(env, "videos/vid_"+str(file), force=True)

        state = monitor.reset()
        i=0
        done = False
        while not done and i<len(actions):
            next_state, reward, done, info= monitor.step(actions[i,:].flatten())
            i+=1
        monitor.close()


if __name__ == "__main__" :
    main()
