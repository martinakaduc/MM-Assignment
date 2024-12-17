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

def train_policy(env, policy, num_episodes=100):
    evaluator = EpisodeEvaluator()
    log_file = open('training_log.txt', 'w')
    
    def log_info(message, console=True):
        log_file.write(message + '\n')
        log_file.flush()
        if console:
            print(message)

    observation, info = env.reset(seed=42)
    log_info("Initial info: " + str(info))

    # Training variables
    ep = 0
    best_reward = float('-inf')
    total_steps = 0

    try:
        while ep < num_episodes and not should_exit:
            episode_reward = 0
            step = 0
            final_observation = None
            
            # Training phase
            while True and not should_exit:
                action = policy.get_action(observation, info)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                calculated_reward = policy.calculate_reward(action, observation, info)
                episode_reward += calculated_reward
                policy.update_policy(calculated_reward, terminated or truncated)
                
                final_observation = next_observation
                observation = next_observation
                step += 1
                total_steps += 1

                if terminated or truncated:
                    # Only log if all products have been processed
                    remaining_products = sum(prod["quantity"] for prod in observation["products"])
                    if remaining_products == 0:
                        policy.log_episode_summary(
                            steps=step,
                            filled_ratio=info['filled_ratio'],
                            episode_reward=episode_reward,
                            observation=observation
                        )
                    
                    # Save model if it's the best so far
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        log_info("\nNew best reward achieved! Saving model...")
                        if isinstance(policy, ProximalPolicyOptimization):
                            policy.save_model("model_ppo_best")
                        elif isinstance(policy, ActorCriticPolicy2):
                            policy.save_model("model_a2c_best")
                    
                    # Save model every 10 episodes
                    if ep > 0 and ep % 10 == 0:
                        log_info(f"\nSaving model at episode {ep}...")
                        if isinstance(policy, ProximalPolicyOptimization):
                            policy.save_model("model_ppo_best")
                        elif isinstance(policy, ActorCriticPolicy2):
                            policy.save_model("model_a2c_best")
                    
                    observation, info = env.reset(seed=ep)
                    ep += 1
                    break
            
            eval_data = {
                'episode_number': ep,
            }

            ep_summary = evaluator.evaluate_episode(final_observation, eval_data)
            print(ep_summary)

            # Evaluation phase every 10 episodes
            if ep > 0 and ep % 10 == 0:
                log_info("\nRunning evaluation phase...")
                policy.training = False  # Switch to evaluation mode
                
                # Run single evaluation episode
                eval_observation, eval_info = env.reset(seed=1000)
                eval_data = {
                    'episode_number': ep,
                    'steps': 0,
                    'total_reward': 0
                }
                
                while True:
                    eval_action = policy.get_action(eval_observation, eval_info)
                    eval_next_obs, _, eval_terminated, eval_truncated, eval_info = env.step(eval_action)
                    
                    # Calculate reward using policy's method
                    eval_reward = policy.calculate_reward(eval_action, eval_observation, eval_info)
                    eval_data['total_reward'] += eval_reward
                    eval_data['steps'] += 1
                    
                    eval_observation = eval_next_obs
                    
                    if eval_terminated or eval_truncated:
                        # Get evaluation summary using evaluator
                        summary = evaluator.evaluate_episode(eval_observation, eval_data)
                        print(summary)
                        log_info(summary)
                        break
                
                policy.training = True  # Switch back to training mode
        evaluator.plot_metrics()
    except Exception as e:
        log_info(f"Error occurred: {str(e)}")
        policy.save_model(f"model_ppo_error_step_{total_steps}")
        raise e
    finally:
        log_file.close()

if __name__ == "__main__":
    # Constants
    # Create environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )
    
    POLICY_TYPE = "ppo" #ppo, a2c
    NUM_EPISODES = 100

    if POLICY_TYPE == "a2c":
        policy = ActorCriticPolicy2()
        train_func = train_policy
    elif POLICY_TYPE == "ppo":
        policy = ProximalPolicyOptimization()
        train_func = train_policy
        # Run training and testing
    train_func(
            env=env,
            policy=policy,
            num_episodes=NUM_EPISODES,
    )
    
    env.close()

