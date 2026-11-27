
import numpy as np

class ReplayBuffer:
  def __init__(self, state_dim, action_dim, max_size):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.state_memory = np.zeros((self.mem_size, state_dim), dtype = np.float32)
    self.next_state_memory = np.zeros((self.mem_size, state_dim), dtype = np.float32)
    self.action_memory = np.zeros((self.mem_size, action_dim), dtype = np.float32)
    self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
    self.terminal_memory = np.zeros(self.mem_size, dtype = np.uint8)

  def store_transition(self, state, action, reward, next_state, done):
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.next_state_memory[index] = next_state
    self.terminal_memory[index] = done

    self.mem_cntr += 1

  def sample_batch(self, batch_size = 256):
    max_mem = min(self.mem_cntr, self.mem_size)
    batch = np.random.choice(max_mem, batch_size)
    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    states_ = self.next_state_memory[batch]
    dones = self.terminal_memory[batch]
    return states, actions, rewards, states_, dones
