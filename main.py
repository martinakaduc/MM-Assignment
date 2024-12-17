import gym_cutting_stock
import gymnasium as gym
import time
from student_submissions.s2210xxx.policy2210xxx import FFDHPolicy,  BFDHPolicy

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)

NUM_EPISODES = 10  # Adjust the number of episodes as needed

if __name__ == "__main__":
    
    #This part for FFDH running : Uncomment to run
    # Initialize metrics
    total_waste = 0
    total_filled_ratio = 0
    total_execution_time = 0

    # Test FFDHPolicy
    gd_policy =FFDHPolicy()

    for ep in range(NUM_EPISODES):
        # Reset the environment
        observation, info = env.reset(seed=ep)
        terminated, truncated = False, False
        episode_waste = 0
        episode_filled_ratio = 0

        while not (terminated or truncated):
            # Measure execution time
            start_time = time.time()
            action = gd_policy.get_action(observation, info)
            execution_time = time.time() - start_time

            # Step through the environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Calculate waste and filled ratio
            try:
                total_area = sum(stock.shape[0] * stock.shape[1] for stock in observation["stocks"])
                used_area = sum((stock > 0).sum() for stock in observation["stocks"])
                waste = total_area - used_area
                filled_ratio = used_area / total_area if total_area > 0 else 0

                episode_waste = waste
                episode_filled_ratio = filled_ratio

            except Exception as e:
                print(f"Error calculating waste: {e}")
                print("Stocks:", observation["stocks"])
                waste = "N/A"
                filled_ratio = 0

            # Add metrics for this step
            total_execution_time += execution_time

            # Display per-step output
            print(f"Episode {ep + 1}:")
            print(f"Waste: {waste}")
            print(f"Filled Ratio: {filled_ratio:.2f}")
            print(f"Execution time: {execution_time:.6f} seconds")

        # Accumulate per-episode metrics
        total_waste += episode_waste
        total_filled_ratio += episode_filled_ratio

    # Calculate averages
    avg_waste = total_waste / NUM_EPISODES
    avg_filled_ratio = total_filled_ratio / NUM_EPISODES
    avg_execution_time = total_execution_time / NUM_EPISODES

    # Print final summary
    print("\nSummary:")
    print(f"Execution time: {avg_execution_time:.6f} seconds")
    print(f"Average Waste across {NUM_EPISODES} episodes: {avg_waste:.2f}")
    print(f"Average Filled Ratio across {NUM_EPISODES} episodes: {avg_filled_ratio:.2f}")
    print(f"Average Time across {NUM_EPISODES} episodes: {avg_execution_time:.6f} seconds")


    #This part for BFDH running : Uncomment to run

    # # Initialize metrics
    # total_waste = 0
    # total_filled_ratio = 0
    # total_execution_time = 0

    # # Test BFDHPolicy
    # bfdh_policy = BFDHPolicy()  # Initialize BFDH policy

    # for ep in range(NUM_EPISODES):
    #     # Reset the environment
    #     observation, info = env.reset(seed=ep)
    #     terminated, truncated = False, False
    #     episode_waste = 0
    #     episode_filled_ratio = 0

    #     while not (terminated or truncated):
    #         # Measure execution time
    #         start_time = time.time()
    #         action = bfdh_policy.get_action(observation, info)  # Get action using BFDHPolicy
    #         execution_time = time.time() - start_time

    #         # Step through the environment
    #         observation, reward, terminated, truncated, info = env.step(action)

    #         # Calculate waste and filled ratio
    #         try:
    #             total_area = sum(stock.shape[0] * stock.shape[1] for stock in observation["stocks"])
    #             used_area = sum((stock > 0).sum() for stock in observation["stocks"])
    #             waste = total_area - used_area
    #             filled_ratio = used_area / total_area if total_area > 0 else 0

    #             episode_waste = waste
    #             episode_filled_ratio = filled_ratio

    #         except Exception as e:
    #             print(f"Error calculating waste: {e}")
    #             print("Stocks:", observation["stocks"])
    #             waste = "N/A"
    #             filled_ratio = 0

    #         # Add metrics for this step
    #         total_execution_time += execution_time

    #         # Display per-step output
    #         print(f"Episode {ep + 1}:")
    #         print(f"Waste: {waste}")
    #         print(f"Filled Ratio: {filled_ratio:.2f}")
    #         print(f"Execution time: {execution_time:.6f} seconds")

    #     # Accumulate per-episode metrics
    #     total_waste += episode_waste
    #     total_filled_ratio += episode_filled_ratio

    # # Calculate averages
    # avg_waste = total_waste / NUM_EPISODES
    # avg_filled_ratio = total_filled_ratio / NUM_EPISODES
    # avg_execution_time = total_execution_time / NUM_EPISODES

    # # Print final summary
    # print("\nSummary:")
    # print(f"Execution time: {avg_execution_time:.6f} seconds")
    # print(f"Average Waste across {NUM_EPISODES} episodes: {avg_waste:.2f}")
    # print(f"Average Filled Ratio across {NUM_EPISODES} episodes: {avg_filled_ratio:.2f}")
    # print(f"Average Time across {NUM_EPISODES} episodes: {avg_execution_time:.6f} seconds")

env.close()
