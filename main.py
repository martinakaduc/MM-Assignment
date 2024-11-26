import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2313640.policy2313640 import Policy2313640

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    # render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)
    # print("Initial observation:", observation)
    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    policy2313640 = Policy2313640()
    observation, info = env.reset(seed=42)
    print("Initial observation:", observation)

    for ep in range(NUM_EPISODES):
        print(f"Episode {ep + 1}")
        action = policy2313640.get_action(observation, info)  # Get action from your policy
        print("Action:", action)
        
        if action:
            # Apply the action in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            print("Reward:", reward, "Filled ratio:", info.get("filled_ratio", 0))
            
            if terminated or truncated:
                print("Episode ended. Resetting environment...")
                observation, info = env.reset(seed=ep)
        else:
            print("No valid action generated. Skipping...")
            break

env.close()
