from policy import Policy
from .floorceil import FloorCeil
from .genetic import GeneticPolicy


class Policy2352819_2211073_2352455_2352137_2353031(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = FloorCeil()
        elif policy_id == 2:
            self.policy = GeneticPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
