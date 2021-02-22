""""
Code adapted and expanded from the original Udacity code project.
"""

import copy
import random
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, hyperparameters, seed):
        """Initialize parameters and noise process."""
        self.mu = hyperparameters.MU * np.ones(size)
        self.theta = hyperparameters.THETA
        self.sigma = hyperparameters.SIGMA
        self.use_sigma_decay = hyperparameters.USE_SIGMA_DECAY
        self.sigma_min = hyperparameters.SIGMA_MIN
        self.sigma_decay = hyperparameters.SIGMA_DECAY
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min if decay is enabled"""
        if self.use_sigma_decay:
            self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state