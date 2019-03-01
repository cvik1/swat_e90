"""
hello.py
hello world demo for Professor Zucker
"""

import gym
import roboschool

def main():

    env_name = "RoboschoolAnt-v1"
    env = gym.make(env_name)

    env.reset()

    for i in range(5000):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        env.render()
        if done:
            break

    print("Demo done!")


if __name__ == "__main__":
    main()
