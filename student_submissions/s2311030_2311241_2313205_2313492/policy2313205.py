import torch
from student_submissions.s2311030_2311241_2313205_2313492.model import PPO
from policy import Policy


class Policy2313205(Policy):
    """
    RL-based Policy for interacting with the CuttingStock environment.
    """

    def __init__(self, actor_path="student_submissions/s2311030_2311241_2313205_2313492/actor_ppo_checkpoint.pth",
                 critic_path="student_submissions/s2311030_2311241_2313205_2313492/critic_ppo_checkpoint.pth", device=None):
        """
        Initialize the RL policy with the trained model.

        Args:
            actor_path (str): Path to the actor model checkpoint.
            critic_path (str): Path to the critic model checkpoint.
            device (str): Device to use for inference ('cuda' or 'cpu').
        """
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing RLPolicy on device: {self.device}")

        # Initialize the PPO model
        self.model = PPO(
            num_stock=100,
            max_w=100,
            max_h=100,
            num_product=25,
            hidden_size=128,
            mode="Test",
            device=self.device
        )

        # Load pre-trained weights
        self.load(actor_path, critic_path)

    def load(self, actor_path, critic_path):
        """Load pre-trained actor and critic weights."""
        # Load actor
        actor_checkpoint = torch.load(actor_path, map_location=self.device)
        self.model.policy_net.load_state_dict(actor_checkpoint["actor_net_state_dict"])
        self.model.policy_net.eval()

        # Load critic
        critic_checkpoint = torch.load(critic_path, map_location=self.device)
        self.model.value_net.load_state_dict(critic_checkpoint["value_net_state_dict"])
        self.model.value_net.eval()

        print(f"Loaded weights from:\n - Actor: {actor_path}\n - Critic: {critic_path}")

    def get_action(self, observation, info=None):
        """
        Get an action using the PPO model.

        Args:
            observation (dict): Current state of the environment.
            info (dict): Additional information about the environment.

        Returns:
            dict: Action to perform in the environment.
        """
        return self.model.get_action(observation, info)
    def reset(self, observation, info=None):
        return self.model.model_reset(observation)