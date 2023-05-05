import gym
from gym import spaces
import numpy as np


class OptimalControlEnv(gym.Env):
    def __init__(self, dt=0.01):
        super(OptimalControlEnv, self).__init__()

        # Define the action and state spaces
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Set initial state
        self.state = np.array([1, 2])

        # Set the time step
        self.dt = dt

    def step(self, action):
        # Apply the action and update the state
        x1, x2 = self.state
        u = action[0]

        f = np.array([-x1 + x2, -0.5 * x1 - 0.5 * x2 * np.sin(x1) ** 2])
        g = np.array([0, np.sin(x1)])

        dx = (f + g * u) * self.dt
        self.state = self.state + dx

        # Calculate the reward (negative cost)
        reward = -0.5 * (x1**2 + x2**2 + u**2)

        # Check for the terminal condition (optional)
        done = False

        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.state = np.array([1, 2])
        return self.state

    def render(self, mode="human"):
        # Rendering function (optional)
        pass
