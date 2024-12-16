from policy import Policy
from .src.best_fit import BestFit
from .src.generate_column import GenerateColumn
class Policy2312438_2312410_2311982(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = BestFit()
        elif policy_id == 2:
            self.policy = GenerateColumn()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
