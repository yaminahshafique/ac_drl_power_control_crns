import numpy as np
import torch

def prep_obs(obs, p_max, n):
    """Scale the previous-power block of the state to [0, 1] so it is visible to the network."""
    o = obs.copy()
    o[n:2 * n] /= p_max
    return o

def watts_to_dbm(w):

    w = np.asarray(w, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            w > 0,
            10.0 * np.log10(w / 1e-3),
            -np.inf
        )

def dbm_to_watts(dbm):
    return 1e-3 * (10.0 ** (dbm / 10.0))


def step_powers_w(info, action, env):

    if "transmit_powers" in info:

        return np.asarray(
            info["transmit_powers"],
            dtype=np.float64
        )

    action = np.clip(
        np.asarray(action, dtype=np.float64),
        -1.0,
        1.0
    )

    return ((action + 1.0) / 2.0) * env.Pmax

def is_interference_violated(actual_interference, threshold):

    actual_interference = float(np.asarray(actual_interference).item())

    threshold = float(np.asarray(threshold).item())

    return bool(actual_interference > threshold)

def update_action_std_schedule(agent, progress, START_MAX_ACTION_STD = 0.25, END_MAX_ACTION_STD = 0.10, MIN_ACTION_STD = 0.05):

    progress = float(np.clip(progress, 0.0, 1.0))

    current_max_std = (
        START_MAX_ACTION_STD
        + (END_MAX_ACTION_STD - START_MAX_ACTION_STD)
        * progress
    )

    agent.min_log_std = np.log(MIN_ACTION_STD)
    agent.max_log_std = np.log(current_max_std)

    with torch.no_grad():

        agent.log_std.clamp_(
            min=agent.min_log_std,
            max=agent.max_log_std
        )

    actual_std = torch.exp(
        agent.log_std.detach()
    ).cpu().numpy()

    return current_max_std, actual_std



def forced_full_power_check(env, episodes=3, seed0=50_000):

    violations = 0
    total = 0

    for e in range(episodes):

        env.reset(seed=seed0 + e)
        for _ in range(env.STEPS_PER_RUN):

            action = np.ones(env.ACTION_DIM, dtype=np.float32)

            _, _, terminated, truncated, info = env.step(action)

            violated = is_interference_violated(
                info["interference"],
                env.Ith
            )

            violations += int(violated)
            total += 1

            if terminated or truncated:
                break

    return violations / max(total, 1)




