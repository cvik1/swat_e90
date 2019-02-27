"""
hello.py
hello world demo for Professor Zucker
"""

import gym
import pybulletgym

def main():

    env_name = "AntPyBulletEnv-v0"
    env = gym.make(env_name)

    env.reset()

    for i in range(5000):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        env.render()

    print("Demo done!")


if __name__ == "__main__":
    main()
