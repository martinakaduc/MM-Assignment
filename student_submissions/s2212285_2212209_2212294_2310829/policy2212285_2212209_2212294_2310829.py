from policy import Policy
from .src.Best_Fit_Decreasing import Best_Fit_Decreasing
from .src.Albano_Suppupo import Albano_Suppuno

class Policy2212285_2212209_2212294_2310829(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = Best_Fit_Decreasing()
        elif policy_id == 2:
            self.policy = Albano_Suppuno()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed
