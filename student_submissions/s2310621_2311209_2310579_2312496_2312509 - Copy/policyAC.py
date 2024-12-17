# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from policy import Policy

# class ActorNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ActorNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)

# class CriticNetwork(nn.Module):
#     def __init__(self, input_dim):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class PolicyAC(Policy):
#     def __init__(self):
#         self.actor_network = ActorNetwork(input_dim=10, output_dim=4)  # Adjust input_dim and output_dim as needed
#         self.critic_network = CriticNetwork(input_dim=10)
#         self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=0.001)
#         self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)
#         self.gamma = 0.99  # Discount factor

#     def get_action(self, observation, info):
#         state = self._preprocess_observation(observation)
#         state_tensor = torch.tensor(state, dtype=torch.float32)
#         action_probs = self.actor_network(state_tensor)
#         action = torch.multinomial(action_probs, 1).item()
#         return self._decode_action(action)

#     def _preprocess_observation(self, observation):
#         # Convert observation to a suitable format for the neural network
#         # This is a placeholder implementation
#         return np.array([0] * 10)  # Adjust based on actual observation structure

#     def _decode_action(self, action):
#         # Convert action index to actual action
#         # This is a placeholder implementation
#         return {"stock_idx": 0, "size": [1, 1], "position": (0, 0)}  # Adjust based on actual action structure

#     def update_policy(self, state, action, reward, next_state, done):
#         state_tensor = torch.tensor(state, dtype=torch.float32)
#         next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
#         action_tensor = torch.tensor(action, dtype=torch.int64)
#         reward_tensor = torch.tensor(reward, dtype=torch.float32)
#         done_tensor = torch.tensor(done, dtype=torch.float32)

#         # Update Critic
#         value = self.critic_network(state_tensor)
#         next_value = self.critic_network(next_state_tensor)
#         target = reward_tensor + (1 - done_tensor) * self.gamma * next_value
#         critic_loss = nn.functional.mse_loss(value, target.detach())
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # Update Actor
#         advantage = (target - value).detach()
#         action_probs = self.actor_network(state_tensor)
#         log_prob = torch.log(action_probs[action_tensor])
#         actor_loss = -log_prob * advantage
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from policy import Policy

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyAC(Policy):
    def __init__(self):
        self.actor_network = ActorNetwork(input_dim=10, output_dim=4)  # Adjust input_dim and output_dim as needed
        self.critic_network = CriticNetwork(input_dim=10)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)
        self.gamma = 0.99  # Discount factor
        self.epsilon_clip = 0.2  # PPO clip parameter
        self.c1 = 0.5  # Value function coefficient
        self.c2 = 0.01  # Entropy coefficient

    def get_action(self, observation, info):
        state = self._preprocess_observation(observation)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor_network(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return self._decode_action(action)

    def _preprocess_observation(self, observation):
        # Convert observation to a suitable format for the neural network
        # This is a placeholder implementation
        return np.array([0] * 10)  # Adjust based on actual observation structure

    def _decode_action(self, action):
        # Convert action index to actual action
        # This is a placeholder implementation
        return {"stock_idx": 0, "size": [1, 1], "position": (0, 0)}  # Adjust based on actual action structure

    def update_policy(self, states, actions, rewards, next_states, dones, old_log_probs):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)

        # Compute advantages
        values = self.critic_network(states_tensor)
        next_values = self.critic_network(next_states_tensor)
        targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_values
        advantages = targets - values

        # Update Critic
        critic_loss = nn.functional.mse_loss(values, targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        action_probs = self.actor_network(states_tensor)
        log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1))
        ratios = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() + self.c1 * critic_loss - self.c2 * (action_probs * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()