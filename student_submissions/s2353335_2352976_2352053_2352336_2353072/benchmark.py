import gym_cutting_stock
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
# from policy import GreedyPolicy, RandomPolicy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from policy import GreedyPolicy, RandomPolicy
from ProximalPolicyOptimization import ProximalPolicyOptimization
from A2C import ActorCriticPolicy2
from EpisodeEvaluator import EpisodeEvaluator
import signal
import sys
# Add global variable for graceful shutdown
should_exit = False

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    global should_exit
    print('\nSignal received. Preparing to exit gracefully...')
    should_exit = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def benchmark_policies(env, policies, num_episodes=5, seed=100):
    evaluators = {name: EpisodeEvaluator() for name in policies.keys()}
    
    for policy_name, policy in policies.items():
        print(f"\nTesting {policy_name}...")
        policy.training = False
        log_file = open(f'benchmark_{policy_name}.txt', 'w')
        
        for episode in range(num_episodes):
            observation, info = env.reset(seed=seed + episode)
            episode_data = {
                'episode_number': episode + 1,
                'steps': 0,
                'trimloss': 0
            }
            done = False
            
            while not done:
                action = policy.get_action(observation, info)
                next_obs, reward, terminated, truncated, info = env.step(action)
                observation = next_obs
                episode_data['steps'] += 1
                done = terminated or truncated
            
            # Evaluate episode
            eval_data = {
                'episode_number': episode,
            }
            summary = evaluators[policy_name].evaluate_episode(observation, eval_data)
            log_file.write(summary + "\n")
            print(summary)
        
        log_file.close()
    
    # Plot comparative results
    plt.figure(figsize=(15, 5))
    
    # Plot filled ratios
    plt.subplot(1, 2, 1)
    for name, evaluator in evaluators.items():
        plt.plot(evaluator.history['episode_numbers'], 
                evaluator.history['filled_ratios'], 
                label=name)
    plt.title('Filled Ratio Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Filled Ratio')
    plt.grid(True)
    plt.legend()
    
    # Plot trim losses
    plt.subplot(1, 2, 2)
    for name, evaluator in evaluators.items():
        plt.plot(evaluator.history['episode_numbers'], 
                evaluator.history['trim_losses'], 
                label=name)
    plt.title('Trim Loss Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Trim Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Constants
    NUM_EPISODES = 20

    # Create environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )

    a2c_policy = ActorCriticPolicy2()
    a2c_policy.load_model("MM241-private/saved_models/model_a2c_best.pt")
    
    ppo_policy = ProximalPolicyOptimization()
    ppo_policy.load_model("MM241-private/saved_models/model_ppo_best.pt")

# Create policies dictionary for benchmarking
    policies = {
        'A2C': a2c_policy,
        'PPO': ppo_policy
    }
    benchmark_policies(env, policies, NUM_EPISODES)
    
    env.close()

