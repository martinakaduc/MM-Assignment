import gym_cutting_stock
import gymnasium as gym
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)
from student_submissions.s2211367_2213730_2213682_2213467_2213768.policy2211367_2213730_2213682_2213467_2213768 import Policy2211367_2213730_2213682_2213467_2213768
import pygame
pygame.display.set_mode((800, 800))

from customEnv import CustomCuttingStockEnv, read_testcase_file

env = CustomCuttingStockEnv(render_mode="human")
data = read_testcase_file('.\student_submissions\s2211367_2213730_2213682_2213467_2213768\custom\data.txt')

NUM_EPISODES = 5

if __name__ == "__main__":
    observation, info = env.reset(seed=42, options=data)

    ep = 0
    policy = Policy2211367_2213730_2213682_2213467_2213768(policy_id=1)
    while ep < NUM_EPISODES:
        
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep,options=data)
            ep += 1
    # Reset the environment
    observation, info = env.reset(seed=42)

    policy = Policy2211367_2213730_2213682_2213467_2213768(policy_id=2)
    ep = 0
    while ep < NUM_EPISODES:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
           
            ep += 1

env.close()
