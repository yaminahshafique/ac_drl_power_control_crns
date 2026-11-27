
"""RolloutBuffer class for PPO and A2C"""
class RolloutBuffer:

    def __init__(self):
        self.actions = []       # Pre-tanh Gaussian samples
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.costs = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.costs.clear()
        self.state_values.clear()
        self.is_terminals.clear()
