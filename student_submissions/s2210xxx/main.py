import gym_cutting_stock
import gymnasium as gym
import numpy as np
from policy2213273_2312469_2311744_2310707_2212941 import Policy2213273_2312469_2311744_2310707_2212941

# Number of episodes to run
NUM_EPISODES = 10

# Function to test and evaluate a policy
def test_policy(policy, policy_name, ep, file):
    """
    Runs a test for the given policy and writes the results to a file.
    Args:
        policy: The policy object to test.
        policy_name: The name of the policy.
        ep: The episode number.
        file: The file object to log results.
    """
    # Reset the environment with the episode number as the seed
    observation, info = env.reset(seed=ep)

    # Start timing the execution
    start_time = np.datetime64('now', 'ms')
    # Array to track which stocks have been used
    is_used = [False] * len(observation["stocks"])
    
    while True:
        # Get the next action from the policy
        action = policy.get_action(observation, info)
        # Apply the action and observe the result
        observation, reward, terminated, truncated, info = env.step(action)
        # Mark the selected stock as used
        is_used[action["stock_idx"]] = True
        # Check if the episode has ended
        if terminated or truncated:
            break

    # Calculate the elapsed time for the episode
    elapsed_time = (np.datetime64('now', 'ms') - start_time) / np.timedelta64(1000, 'ms')
    # Evaluate the policy's performance
    area_wasted, area_wasted_ratio = policy.evaluate(observation["stocks"], is_used)
    
    # Log the results to the file
    file.write(f"----{policy_name}---\n")
    file.write(f"\tArea wasted:\t {area_wasted}\n")
    file.write(f"\tArea wasted ratio:\t {area_wasted_ratio}\n")
    file.write(f"\tTime:\t {elapsed_time:.2f}s\n")
    print(f"Episode {ep + 1}, {policy_name}: {elapsed_time:.2f}s")

# Main script
if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("gym_cutting_stock/CuttingStock-v0", render_mode="human")
    
    # Open the result file in append mode
    with open("result.txt", "w") as file:
        for ep in range(NUM_EPISODES):
            # Log the start of the episode
            file.write(f"Episode {ep + 1}-----------------------------\n")
            
            # Test the first policy
            policy_1 = Policy2213273_2312469_2311744_2310707_2212941(policy_id=1)
            test_policy(policy_1, "Dynamic Programming Policy", ep, file)
            
            # Test the second policy
            policy_2 = Policy2213273_2312469_2311744_2310707_2212941(policy_id=2)
            test_policy(policy_2, "Simulated Annealing Policy", ep, file)
    
    # Close the environment after testing
    env.close()