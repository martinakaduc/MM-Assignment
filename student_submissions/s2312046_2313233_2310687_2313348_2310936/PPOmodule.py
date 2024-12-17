'''
Description: Supporting classes for PPO 
author: Lương Minh Thuận  - 2313348
Time: Nov, 2024
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class FilterCNN(nn.Module):
    def __init__(self, out_dim=3):
        super(FilterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, out_dim, kernel_size=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.Conv2d(out_dim, 1, kernel_size=1)
        self.bn2 = nn.LazyBatchNorm2d()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, mask, bias_mask=None):
        x = torch.cat([x.unsqueeze(dim=1), mask.unsqueeze(dim=1), bias_mask.unsqueeze(dim=1)], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, filter_out, alpha, chkpt_dir='./student_submissions/s2312046_2313233_2310687_2313348_2310936/models'):
        super(ActorCritic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'ac_torch_ppo1')
        if not os.path.exists(self.checkpoint_file):
            self.checkpoint_file = os.path.join('./models', 'ac_torch_ppo1')
        self.actor = nn.Conv2d(1, 1, kernel_size=1)
        self.filter = FilterCNN(filter_out)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        stock, bias_mask, mask = state[:, 0], state[:, 1], state[:, 2]
        f = self.filter(stock, mask, bias_mask)
        x = self.actor(f)
        x = x.masked_fill(mask == 0, -1e9)
        x = torch.flatten(x, start_dim=-3)
        x = torch.softmax(x, dim=-1)
        dist = Categorical(x)
        #f = torch.flatten(f, start_dim=-3)
        value = torch.max(f)
        return dist, value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(self.checkpoint_file,weights_only=True,map_location=self.device))
        self.to(self.device)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
