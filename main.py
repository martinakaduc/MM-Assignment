import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2310139_2310090_2310191_2310242_2310423.policy2310139_2310090_2310191_2310242_2310423 import Policy2310139_2310090_2310191_2310242_2310423
import time

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

    # Uncomment the following code to test your policy
    # Reset the environment
#     observation, info = env.reset(seed=42)
#     print(info)

#     policy2210xxx = Policy2310139_2310090_2310191_2310242_2310423()
#     for _ in range(200):
#         action = policy2210xxx.get_action(observation, info)
#         observation, reward, terminated, truncated, info = env.step(action)
#         print(info)

#         if terminated or truncated:
#             observation, info = env.reset()

# env.close()

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

def test_policy(policy, policy_name, env, num_episodes=NUM_EPISODES):
    """Test a given policy on the Cutting Stock environment."""
    print(f"Testing {policy_name}...")

    total_time = 0
    final_filled_ratios = []

    # Run the test 5 times
    for run in range(1):
        observation, info = env.reset(seed=42)
        episode_start_time = time.time()

        # Simulate for num_episodes
        for ep in range(num_episodes):
            action = policy.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset(seed=ep)
        
        episode_end_time = time.time()
        total_time += episode_end_time - episode_start_time
        # Collect the final filled ratio for this run
        final_filled_ratios.append(info['filled_ratio'])

    # Compute the average over the 5 final filled ratios
    avg_filled_ratio = sum(final_filled_ratios) / len(final_filled_ratios)
    print(f"{policy_name} - Average of Final Filled Ratios: {avg_filled_ratio:.4f}")
    print(f"Total Time: {total_time:.4f} seconds")


if __name__ == "__main__":
    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test RandomPolicy
    rd_policy = RandomPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    # observation, info = env.reset(seed=42)
    # print(info)

    # policy2210xxx = Policy2210xxx()
    # for _ in range(200):
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()

env.close()

