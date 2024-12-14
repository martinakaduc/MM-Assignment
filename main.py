import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2012741.policy2012741 import Policy2012741

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)

    policy2012741 = Policy2012741(policy_id=1)
    for _ in range(200):
        action = policy2012741.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)

        if terminated or truncated:
            observation, info = env.reset()

env.close()
