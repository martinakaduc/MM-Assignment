from policy import Policy
from .src.ffd_ma import FFD_MA
from .src.knapsackbased import KnapsackBased

class Policy2310380_2313862_2311824_2310768_2312646(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        
        if policy_id == 1:
            self.policy = FFD_MA()
        elif policy_id == 2:
            self.policy = KnapsackBased()
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
        
    
        
