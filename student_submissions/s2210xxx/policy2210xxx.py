from policy import Policy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from student_submissions.s_2311015_2311464_2311616_2112278_2313327.src.genetic import GeneticPolicy
from student_submissions.s_2311015_2311464_2311616_2112278_2313327.src.RL import ActorCriticPolicy

class Policy2311015_2311464_2311616_2112278_2313327(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = GeneticPolicy()
        elif policy_id == 2:
            self.policy = ActorCriticPolicy()
            pass

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed