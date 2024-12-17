from policy import Policy
from .ProximalPolicyOptimization import ProximalPolicyOptimization
from .A2C import ActorCriticPolicy2

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class Policy2353335_2352976_2352053_2352336_2353072(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.training = False
        self.policy = None
        if policy_id == 1:
            self.policy = ActorCriticPolicy2()
            self.policy.load_model('saved_models/model_a2c_best.pt')
        elif policy_id == 2:
            self.policy = ProximalPolicyOptimization()
            self.policy.load_model('saved_models/model_a2c_best.pt')
        self.policy.training = False
            

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)