from policy import Policy;
from student_submissions.s2352378_2352977_2252100_2352012_2352130 import algorithms;

class Policy2352378_2352977_2252100_2352012_2352130(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = algorithms.BranchAndBound();
        if policy_id == 2:
            self.policy = algorithms.FirstFitDecreasingHeight();

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)