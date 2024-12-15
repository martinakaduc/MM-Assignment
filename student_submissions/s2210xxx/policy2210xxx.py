from policy import Policy
from student_submissions.s2210xxx.policy2352921 import Policy2352921_2353283_2352129_2352659_2352948_1, Policy2352921_2353283_2352129_2352659_2352948_2

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = Policy2352921_2353283_2352129_2352659_2352948_1()
        elif policy_id == 2:
            self.policy = Policy2352921_2353283_2352129_2352659_2352948_2()
            pass

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)
    
    # Student code here
    # You can add more functions if needed