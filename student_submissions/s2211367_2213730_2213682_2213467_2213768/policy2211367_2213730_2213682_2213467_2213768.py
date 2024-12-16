from policy import Policy
from .policies.policy1 import CornerPoint
from .policies.policy2 import BestFit
import numpy as np


class Policy2211367_2213730_2213682_2213467_2213768(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        if policy_id == 1:
            self.policy = CornerPoint()
        elif policy_id == 2:
            self.policy = BestFit()
             
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)