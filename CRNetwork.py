import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CRNetwork(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, n_ST=2, noise=1e-6, int_thresh=-28.0,
        max_power=10e-3, steps_per_run=100, dtype=np.float32, seed=None,
        interference_reference_w=1.0, error_variance=0.1, hist_len = 1,
        reward_mode="paper", alpha=0.2,beta=10.0):
        super().__init__()

        if not isinstance(n_ST, (int, np.integer)) or n_ST < 1:
            raise ValueError("n_ST must be a positive integer.")
        if not isinstance(steps_per_run, (int, np.integer)) or steps_per_run < 1:
            raise ValueError("steps_per_run must be a positive integer.")
        if not np.issubdtype(np.dtype(dtype), np.floating):
            raise ValueError("Use a floating-point observation dtype.")

        parameters = (noise, int_thresh, max_power, interference_reference_w)
        if not np.all(np.isfinite(parameters)):
            raise ValueError("Environment parameters must be finite.")
        if noise <= 0 or max_power <= 0 or interference_reference_w <= 0:
            raise ValueError("Noise, power budget, and interference reference must be positive.")
        if not np.isfinite(error_variance) or error_variance < 0:
            raise ValueError("error_variance must be finite and nonnegative.")
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative.")

        self.k = int(n_ST)
        self.N0 = float(noise)
        self.Pmax = float(max_power)
        self.Ith = float(10**(int_thresh / 10))

        if not np.isfinite(self.Ith) or self.Ith <= 0:
            raise ValueError("Converted interference threshold must be finite and positive.")

        self.max_steps = int(steps_per_run)
        self.dtype = np.dtype(dtype)

        self.hist_len = int(hist_len)
        assert self.hist_len >= 1
        self.obs_dim = (self.hist_len + 1) * self.k + 1
        self.action_dim = self.k

        self.error_variance = float(error_variance)
        self.beta = float(beta)
        self.alpha = float(alpha)

        self.reward_mode = "paper" if reward_mode == "paper" else "custom"

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=self.dtype
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.k,),
            dtype=self.dtype
        )

        # Random number generator
        self.np_random = np.random.default_rng(seed)
        self.step_count = 0
        self._needs_reset = True

    def get_channel_gain(self, update=True):

        # Update complex channel coefficients only when requested
        if update:
           # Noisy estimate
            w_p = (self.np_random.standard_normal(self.k) + 1j * self.np_random.standard_normal(self.k)) / np.sqrt(2)
            w_s = (self.np_random.standard_normal(self.k)+ 1j * self.np_random.standard_normal(self.k)) / np.sqrt(2)

            # Temporally correlated ST-PR channels
            self.hint = ((1 - self.alpha) * self.hint + np.sqrt(2 * self.alpha - self.alpha**2) * w_p)
            self.hsr = ((1 - self.alpha) * self.hsr + np.sqrt(2 * self.alpha - self.alpha**2) * w_s)
        # Actual channel power gains
        g_pr_actual = np.abs(self.hint)**2
        g_sr = np.abs(self.hsr)**2
        # Imperfect CSI on ST-PR links only
        error = np.sqrt(self.error_variance / 2) * (self.np_random.standard_normal(self.k) + 1j * self.np_random.standard_normal(self.k))

        hpr_est = self.hint + error
        gpr_est = np.abs(hpr_est)**2
        self.hpr_est = hpr_est

        return g_pr_actual, gpr_est, g_sr

    def _observation(self):
        # return np.concatenate(
        #     [self.hint_est, self.P_prev, [self.r_prev]]
        # ).astype(self.dtype)
        return np.concatenate([self.est_hist.ravel(), self.P_prev, [self.r_prev]]).astype(self.dtype)

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0
        self.P_prev = np.zeros(self.k)
        self.r_prev = 0.0

        # Initialize complex channel coefficients
        self.hint = (self.np_random.standard_normal(self.k) + 1j * self.np_random.standard_normal(self.k)) / np.sqrt(2)

        self.hsr = (self.np_random.standard_normal(self.k) + 1j * self.np_random.standard_normal(self.k)) / np.sqrt(2)

        # Initial channel gains and imperfect CSI
        self.g_pr_actual, self.hint_est, self.g_sr = (self.get_channel_gain(update=False))
        self.est_hist = np.tile(self.hint_est, (self.hist_len, 1))   # pad history with the current estimate

        self._needs_reset = False

        return self._observation(), {}

    def step(self, action):

        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping or after an episode ends.")

        action = np.asarray(action, dtype=np.float64)

        if action.shape != (self.k,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected a finite action with shape ({self.k},).")

        raw_action = np.clip(action, -1.0, 1.0)

        # Action-to-power mapping
        fractions = (raw_action + 1.0) / 2.0
        powers = fractions * self.Pmax

        # Capacity using actual desired channel gains
        signal_power = powers * self.g_sr
        snr = signal_power / self.N0
        capacity = np.log2(1.0 + snr)

        # Actual primary interference
        interference = float(np.sum(powers * self.g_pr_actual))

        normalized_violation = max(
            0.0,
            (interference - self.Ith) / self.Ith
        )

        power_ok = bool(
            np.all(powers >= 0) and np.all(powers <= self.Pmax)
        )

        feasible = bool(interference <= self.Ith and power_ok)

        # Reward calculation
        if self.reward_mode == "paper":
            reward = float(capacity.sum()) if feasible else -interference
        else:
            reward = float(capacity.sum()) - self.beta * normalized_violation

        info = {
            "capacity": capacity.copy(),
            "snr": snr.copy(),
            "interference": interference,
            "constraints_satisfied": feasible,
            "allocation_fractions": fractions.copy(),
            "transmit_powers": powers.copy(),
            "actual_interference_gains": self.g_pr_actual.copy(),
            "desired_gains": self.g_sr.copy(),
            "estimated_interference": float(np.sum(powers * self.hint_est)),
            "normalized_excess": float(normalized_violation),
            "penalty": float(self.beta * normalized_violation),
            "actual_violation": bool(interference > self.Ith),
            "estimated_violation": bool(np.sum(powers * self.hint_est) > self.Ith),
        }
        # Store previous action and reward
        self.P_prev = powers.copy()
        self.r_prev = reward

        self.step_count += 1

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        # Generate next channel realization for the next observation
        self.g_pr_actual, self.hint_est, self.g_sr = (self.get_channel_gain(update=True))
        self.est_hist = np.vstack([self.hint_est, self.est_hist[:-1]])  # newest first

        self._needs_reset = truncated

        return self._observation(), reward, terminated, truncated, info
