from policy import Policy


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        pass

    # Student code here
    # You can add more functions if needed