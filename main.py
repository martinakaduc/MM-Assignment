import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2250013.policy2250013 import Policy2250013

from time import perf_counter

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
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # Reset the environment
    SEED = 10
    info_list = []
    time_list = []
    
    for s in range(SEED):
        print('\t => ', s)
        observation, info = env.reset(seed=s)
        print(info)

        start = perf_counter()
        # policy2210xxx = GreedyPolicy()
        # policy2210xxx = RandomPolicy()
        policy2210xxx = Policy2250013(policy_id=2)
        for i in range(NUM_EPISODES):
            action = policy2210xxx.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            print('\t', i, info)
            
            if terminated or truncated:
                observation, info = env.reset()
        
        print('-' * 20)
        t = perf_counter() - start
        time_list.append(t)
        info_list.append(info)
        print(t, 's')
        print(info)
        print('-' * 20)
    
    print()
    print('=' * 30)
    print('=' * 30)
    print()
    
    print(info_list)
    print(time_list)
    total_trim_loss = sum([info['trim_loss'] for info in info_list])
    print('total_trim_loss: ', total_trim_loss)
    print('total time: ', sum(time_list))
env.close()
