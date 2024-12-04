# import gym_cutting_stock
# import gymnasium as gym
# from policy import GreedyPolicy, RandomPolicy
# from student_submissions.s2210xxx.policy2210xxx import GeneticPolicy

# # Create the environment
# env = gym.make(
#     "gym_cutting_stock/CuttingStock-v0",
#     render_mode="human",  # Comment this line to disable rendering
# )
# NUM_EPISODES = 100

# if __name__ == "__main__":
#     # # Reset the environment
#     # observation, info = env.reset(seed=42)

#     # # Test GreedyPolicy
#     # gd_policy = GreedyPolicy()
#     # ep = 0
#     # while ep < NUM_EPISODES:
#     #     action = gd_policy.get_action(observation, info)
#     #     observation, reward, terminated, truncated, info = env.step(action)

#     #     if terminated or truncated:
#     #         observation, info = env.reset(seed=ep)
#     #         print(info)
#     #         ep += 1

#     # # Reset the environment
#     # observation, info = env.reset(seed=42)

#     # # Test RandomPolicy
#     # rd_policy = RandomPolicy()
#     # ep = 0
#     # while ep < NUM_EPISODES:
#     #     action = rd_policy.get_action(observation, info)
#     #     observation, reward, terminated, truncated, info = env.step(action)

#     #     if terminated or truncated:
#     #         observation, info = env.reset(seed=ep)
#     #         print(info)
#     #         ep += 1

#     # Uncomment the following code to test your policy
#     # Reset the environment
#     observation, info = env.reset(seed=42)
#     print(info)

#     policy2210xxx = GeneticPolicy()
#     for _ in range(200):
#         action = policy2210xxx.get_action(observation, info)
#         observation, reward, terminated, truncated, info = env.step(action)
#         print(info)

#         if terminated or truncated:
#             observation, info = env.reset()

# env.close()

import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import GeneticPolicy

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

def test_policy(policy, policy_name, env, num_episodes=NUM_EPISODES):
    """Test a given policy on the Cutting Stock environment."""
    observation, info = env.reset(seed=42)
    print(f"Testing {policy_name}...")

    filled_ratios = []

    for ep in range(num_episodes):
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        filled_ratios.append(info['filled_ratio'])
        
        if terminated or truncated:
            observation, info = env.reset(seed=ep)

    avg_filled_ratio = sum(filled_ratios) / len(filled_ratios)
    print(f"{policy_name} - Average Filled Ratio: {avg_filled_ratio:.4f}")

if __name__ == "__main__":
    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    test_policy(gd_policy, "GreedyPolicy", env)

    # Test RandomPolicy
    rd_policy = RandomPolicy()
    test_policy(rd_policy, "RandomPolicy", env)

    # Test GeneticPolicy
    policy2210xxx = GeneticPolicy()
    test_policy(policy2210xxx, "GeneticPolicy", env)

env.close()