class RCPO:
    def __init__(self, alpha=0.25, lambda_init=0.0, lambda_lr=0.01,):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        if lambda_init < 0:
            raise ValueError("lambda_init must be non-negative")

        if lambda_lr <= 0:
            raise ValueError("lambda_lr must be positive")

        self.alpha = float(alpha)
        self.lagrange_multiplier = float(lambda_init)
        self.lambda_lr = float(lambda_lr)

        self.episode_cost = 0.0
        self.episode_steps = 0

    def compute_cost(self, actual_interference, threshold):
        """Return 1 if the interference constraint is violated."""
        return float(actual_interference > threshold)

    def modify_reward(self, reward, cost):
        """Apply the current RCPO penalty to the original reward."""
        return float(reward - self.lagrange_multiplier * cost)

    def record_cost(self, cost):
        """Accumulate constraint costs during an episode."""
        self.episode_cost += float(cost)
        self.episode_steps += 1

    def end_episode(self):
        """Calculate episode violation rate and update lambda."""
        if self.episode_steps == 0:
            raise RuntimeError("Cannot end an empty episode")

        violation_rate = self.episode_cost / self.episode_steps

        self.lagrange_multiplier = max(
            0.0,
            self.lagrange_multiplier
            + self.lambda_lr * (violation_rate - self.alpha)
        )

        result = {
            "violation_rate": violation_rate,
            "lambda": self.lagrange_multiplier,
        }

        self.reset_episode()
        return result

    def reset_episode(self):
        """Clear episode statistics without resetting lambda."""
        self.episode_cost = 0.0
        self.episode_steps = 0
