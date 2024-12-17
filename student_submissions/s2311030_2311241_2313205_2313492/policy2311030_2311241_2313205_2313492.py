from student_submissions.s2311030_2311241_2313205_2313492.policy2311241 import Policy2311241
from student_submissions.s2311030_2311241_2313205_2313492.policy2313492 import Policy2313492
from student_submissions.s2311030_2311241_2313205_2313492.policy2313205 import Policy2313205
from policy import Policy


class Policy2311030_2311241_2313205_2313492(Policy):
    def __init__(self, policy_id=1, **kwargs):
        """
        Initialize the wrapper policy based on the selected policy ID.

        Args:
            policy_id (int): ID of the policy to use (1, 2, or 3).
            kwargs: Additional arguments for policy initialization.
        """
        assert policy_id in [1, 2, 3], "Policy ID must be 1, 2, or 3"
        self.policy_id = policy_id

        # Initialize the appropriate policy
        if policy_id == 1:
            self.policy = Policy2311241(**kwargs)
        elif policy_id == 2:
            self.policy = Policy2313492(**kwargs)
        elif policy_id == 3:
            self.policy = Policy2313205(**kwargs)

    def get_action(self, observation, info):
        """
        Get an action from the selected policy.

        Args:
            observation (dict): Environment observation.
            info (dict): Additional environment info.

        Returns:
            dict: Action chosen by the policy.
        """
        return self.policy.get_action(observation, info)
    # def reset(self, observation, info):
    #     if (self.policy_id == 3):
    #         return self.policy.reset(observation)
    #     else:
    #         return