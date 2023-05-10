import gym
from gym import spaces
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class OptimalControlEnv(gym.Env):
    def __init__(self, dt=1):
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
        
        # Define the desired state for LQR
        self.desired_state = np.array([0, 0])

        # Calculate the LQR gain matrix
        A, B = self._get_linearized_system()
        Q = np.diag([1, 1]) # State cost matrix
        R = 0.1 # Control cost
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)

    def _get_linearized_system(self):
        # Get the linearized system dynamics around the current state
        x1, x2 = self.state
        u = 0 # Zero input
        A = np.array([[-1, 1], [-0.5 - np.sin(x1)**2 * x2 / 2, -0.5 * np.sin(x1)**2]])
        B = np.array([[0], [np.sin(x1)]])
        return A, B

    def step(self, action):
        # Apply the LQR control law to the current state
        x_deviation = self.state - self.desired_state
        u = -self.K.dot(x_deviation)

        # Apply the action and update the state
        x1, x2 = self.state
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

def LQR():

    # Create the environment
    env = OptimalControlEnv(dt=0.01)

    rewards = []
    num_episodes = 1000
    # Run the simulation with LQR control
    x1_list = []
    x2_list = []
    u_list = []
    state = env.reset()
    for i in range(num_episodes):
        action = -np.matmul(env.K, (state - env.desired_state))
        state, reward, done, _ = env.step(action)
        x1_list.append(state[0])
        x2_list.append(state[1])
        u_list.append(action[0])
        rewards.append(reward)
        if done:
            break

    # Get the final value of u
    final_u = action[0]
    print("Final value of u:", final_u)

    # Plots
    # time_steps = np.arange(len(x1_list)) * env.dt
    # plot_state_trajectory(x1_list, x2_list, time_steps)
    # plot_u_vs_time(time_steps, u_list)
    # plot_rewards_vs_time(rewards)


def plot_state_trajectory(x1_list, x2_list, time_steps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up the axes labels
    ax.set_zlabel('Time (s)')
    ax.set_ylabel('x1')
    ax.set_xlabel('x2')

    # Plot the state trajectory
    ax.plot(x1_list, x2_list, time_steps)

    # Show the plot
    plt.show()

def plot_rewards_vs_time(rewards):
    # Plot the rewards over time
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

def plot_u_vs_time(time_steps, u_list):
    plt.figure()
    plt.plot(time_steps, u_list)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (u)')
    plt.title('Control Input vs Time')
    plt.show()

LQR()

