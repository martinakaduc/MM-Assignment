from policy import Policy
import numpy as np
from scipy.optimize import linprog
from student_submissions.s2211315_2213705_2211571.CG_algorithm import ColunmGeneraton
from student_submissions.s2211315_2213705_2211571.Genetic_algorithm import Genetic

class Policy2211315_2213705_2211571(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = ColunmGeneraton()
        elif policy_id == 2:
            self.policy =  Genetic()

    def get_action(self, observation, info):
        if self.policy:
            return self.policy.get_action(observation, info)
        else:
            return {"stock_idx": -1, "size": [0, 0], "position": [0, 0]}  # Default action

    # Student code here
    # You can add more functions if needed





