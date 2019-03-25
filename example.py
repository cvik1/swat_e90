"""
example.py
"""

import numpy as np
import gym
from gym.wrappers import Monitor
import roboschool

# make the ant environment
env = gym.make("RoboschoolAnt-v1")
# make a monitor to record the video
monitor = Monitor(env, "randomAnt", force=True)
monitor.reset()

# run one episode of the random agent on Ant

done = False
while not done:
    _, _, done, _ = monitor.step(env.action_space.sample())
