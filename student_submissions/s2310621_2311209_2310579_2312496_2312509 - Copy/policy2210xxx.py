from policy import Policy
from policyHeuristic import PolicyHeuristic
from policyAC import PolicyAC


class Policy2310621_2311209_2310579_2312496_2312509(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.polici_id = 1
        elif policy_id == 2:
            self.policy_id = 2

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            heuristic = PolicyHeuristic(self.policy_id)
            return heuristic.get_action(observation, info)
        if self.policy_id == 2:
            AC = PolicyAC()
            return AC.get_action(observation, info)
