"""TD-3 class
Note: AI tools were used in debugging, especially with:
 device mismatch errors
 performing checks
 printing error warnings
 debugging load and save functions
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class TD3_Agent:
    def __init__(self, alpha, beta, state_dim, tau, action_max,
                 gamma=0.99, update_actor_interval=2, action_dim=2,
                 max_size=1_000_000, batch_size=256, seed=0, device=None,
                 policy_noise=0.2, noise_clip=0.5):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("Require gamma in [0, 1] and tau in (0, 1].")
        if not isinstance(update_actor_interval, (int, np.integer)) or update_actor_interval < 1:
            raise ValueError("update_actor_interval must be a positive integer.")
        if batch_size < 1 or max_size < batch_size:
            raise ValueError("Require max_size >= batch_size >= 1.")
        if (not np.isfinite(policy_noise) or not np.isfinite(noise_clip)
                or policy_noise < 0 or noise_clip < 0):
            raise ValueError("Noise settings must be finite and nonnegative.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_actor_iter = update_actor_interval
        self.delay_counter = 0

        self.high = np.broadcast_to(
            np.asarray(action_max, dtype=np.float32), (action_dim,)
        ).copy()
        if not np.all(np.isfinite(self.high)) or np.any(self.high <= 0):
            raise ValueError("action_max must contain positive finite raw-action bounds.")
        self.low = -self.high
        self.max_action = self.high.copy()
        
        self.high_t = torch.as_tensor(self.high, device=self.device)
        self.low_t = torch.as_tensor(self.low, device=self.device)
        self.policy_noise = policy_noise * self.high_t
        self.noise_clip = noise_clip * self.high_t

        torch.manual_seed(seed)
        np.random.seed(seed)  
        self.rng = np.random.default_rng(seed)
        self.memory = ReplayBuffer(state_dim, action_dim, max_size)
        if not np.issubdtype(self.memory.action_memory.dtype, np.floating):
            raise TypeError("ReplayBuffer.action_memory must use a floating-point dtype.")

        self.actor = Actor(state_dim, action_dim, self.high.copy()).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)

        # Exact initial copies; targets never receive gradient updates
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic_1 = deepcopy(self.critic_1).requires_grad_(False)
        self.target_critic_2 = deepcopy(self.critic_2).requires_grad_(False)
        for network in (self.actor, self.critic_1, self.critic_2,
                        self.target_actor, self.target_critic_1, self.target_critic_2):
            network.device = self.device
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_1.optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=beta)
        self.critic_2.optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=beta)

    def choose_action(self, s, noise_std=0.0, *, exploration_noise=None):
        
        if exploration_noise is not None:
            if noise_std != 0.0:
                raise ValueError("Specify only noise_std or exploration_noise.")
            noise_std = exploration_noise
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("Exploration noise must be finite and nonnegative.")
        s = np.asarray(s, dtype=np.float32)
        if s.shape != (self.state_dim,) or not np.all(np.isfinite(s)):
            raise ValueError(f"Expected a finite observation with shape ({self.state_dim},).")
        state = torch.as_tensor(s, device=self.device).unsqueeze(0)
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train(was_training)
        if noise_std > 0:
            action = action + self.rng.normal(size=self.action_dim) * noise_std * self.high
        return np.clip(action, self.low, self.high).astype(np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, float(reward), next_state, bool(done))

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None

        states, actions, rewards, next_states, done = self.memory.sample_batch(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1, 1)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

        # Target Q
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(min=-self.noise_clip, max=self.noise_clip)
            next_actions = self.target_actor(next_states) + noise
            next_actions = next_actions.clamp(min=self.low_t, max=self.high_t)
            next_q1 = self.target_critic_1(next_states, next_actions)
            next_q2 = self.target_critic_2(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - done) * torch.minimum(next_q1, next_q2)

        # Critic update
        q1_loss = F.mse_loss(self.critic_1(states, actions), target_q)
        q2_loss = F.mse_loss(self.critic_2(states, actions), target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.delay_counter += 1
        actor_loss_value = float("nan")

        # Actor and target nets update
        if self.delay_counter % self.update_actor_iter == 0:
            self.critic_1.requires_grad_(False)
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.critic_1.requires_grad_(True)
            actor_loss_value = actor_loss.item()
            self.update_network_parameters()

        # return {
        #     "actor_loss": actor_loss_value,
        #     "critic_1_loss": q1_loss.item(),
        #     "critic_2_loss": q2_loss.item(),
        #     "critic_loss": critic_loss.item(),  # Sum of the two MSE losses.
        #     "actor_updated": actor_loss_value is not None,
        #     "update_step": self.delay_counter,
        # }
        return [actor_loss_value, critic_loss.item()]

    @torch.no_grad()
    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else tau
        if not 0 <= tau <= 1:
            raise ValueError("tau must lie in [0, 1].")
        pairs = ((self.actor, self.target_actor),
                 (self.critic_1, self.target_critic_1),
                 (self.critic_2, self.target_critic_2))
        for online, target in pairs:
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.mul_(1 - tau).add_(parameter, alpha=tau)

    def save_models(self, path="td3_models.pt"):
        """Save weights for evaluation, not a complete training-resume state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        networks = ("actor", "critic_1", "critic_2", "target_actor",
                    "target_critic_1", "target_critic_2")
        checkpoint = {name: getattr(self, name).state_dict() for name in networks}
        checkpoint.update(state_dim=self.state_dim, action_dim=self.action_dim,
                          high=self.high.tolist())
        torch.save(checkpoint, path)

    def load_models(self, path="td3_models.pt"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if (checkpoint["state_dim"] != self.state_dim
                or checkpoint["action_dim"] != self.action_dim
                or not np.allclose(checkpoint["high"], self.high)):
            raise ValueError("Checkpoint dimensions or raw-action bounds do not match.")
        for name in ("actor", "critic_1", "critic_2", "target_actor",
                     "target_critic_1", "target_critic_2"):
            network = getattr(self, name)
            network.load_state_dict(checkpoint[name])
            network.eval()
