import numpy as np
import random
import copy

"""Ornstein-Uhlenbeck Noise (encourage exploration of actions)"""
class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros_like(mean)
        self.x_prev = np.copy(self.x0)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0
# class OU_noise(object):
#     """
#     size: size of noise vector to be generated (equivalent to number of actions)
#     mu: mean of the noise, 0 as default
#     sigma: noise volatility, controls the magnitude of fluctuations
#     """

#     def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.seed = random.seed(seed)
#         self.reset()

#     def reset(self):
#         """Reset internal state (noise) to mean"""
#         self.state = copy.copy(self.mu)
    
#     def sample(self):
#         """Update internal state and return as a noise sample.
#         Uses the current state of the noise and generates the next sample
#         """

#         dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for i in range(len(self.state))])
#         self.state += dx

#         # copy of state to modify for action-specific ranges
#         noisy_actions = np.array(self.state)

#         # scale noise for steer action (second action aka index 1)
#         noisy_actions[1] *= 1.0

#         # scale and shift noise for brake and throttle which are positive only
#         # throttle is first action (index 0) and brake is third action (index 2)
#         noisy_actions[[0, 2]] = noisy_actions[[0, 2]] * 0.5 + 0.5 # scales the noise to [0, 1]

#         # ensure action values are in their correct ranges
#         noisy_actions[0] = np.clip(noisy_actions[0], 0, 1) # throttle
#         noisy_actions[1] = np.clip(noisy_actions[1], -1, 1) # steer
#         noisy_actions[2] = np.clip(noisy_actions[2], 0, 1) # brake
        
#         return noisy_actions
    
    