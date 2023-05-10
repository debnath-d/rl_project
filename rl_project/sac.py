import torch
import torch.nn as nn
import torch.optim as optim

# from torch.distributions import Normal

from env import OptimalControlEnv

import matplotlib.pyplot as plt

# import seaborn as sns
from tqdm import tqdm
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = torch.relu(self.layer_1(torch.cat([x, u], 1)))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = deque(maxlen=int(max_size))

    def add(self, state, action, next_state, reward, done):
        self.storage.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return state, action, next_state, reward, done


class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.alpha = 0.2
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(1 - done).to(device).unsqueeze(1)

        # Critic training
        next_action = self.actor_target(next_state)
        noise = (
            torch.FloatTensor(action).data.normal_(0, 0.1 * self.max_action).to(device)
        )
        noise = torch.clamp(noise, -0.5 * self.max_action, 0.5 * self.max_action)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        target_q1 = self.critic_target_1(next_state, next_action)
        target_q2 = self.critic_target_2(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (done * 0.99 * target_q).detach()

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss1 = nn.MSELoss()(current_q1, target_q)
        critic_loss2 = nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer_1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer_2.step()

        # Actor training
        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(
            self.critic_1.parameters(), self.critic_target_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic_2.parameters(), self.critic_target_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(
            self.critic_optimizer_1.state_dict(), filename + "_critic_optimizer_1"
        )
        torch.save(
            self.critic_optimizer_2.state_dict(), filename + "_critic_optimizer_2"
        )

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
        self.critic_optimizer_1.load_state_dict(
            torch.load(filename + "_critic_optimizer_1")
        )
        self.critic_optimizer_2.load_state_dict(
            torch.load(filename + "_critic_optimizer_2")
        )


def train(env, num_episodes, max_timesteps, batch_size):
    reward_history = []
    state_trajectory = []
    control_input_history = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)

            if len(replay_buffer.storage) >= batch_size:
                agent.train(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward
            state_trajectory.append(state)
            control_input_history.append(action)

            if done:
                break

        reward_history.append(episode_reward)
        # print(f"Episode {episode + 1}: Reward = {episode_reward}")

        return reward_history, state_trajectory, control_input_history


def plot_rewards(total_rewards):
    # Plot rewards
    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards vs Episodes")
    plt.savefig("rewards.png")


def plot_state_trajectory(state_trajectory):
    state_trajectory = np.array(state_trajectory)
    timesteps = np.arange(state_trajectory.shape[0])

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("State Trajectory")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("$x_1(t)$")
    ax.set_zlabel("$x_2(t)$")

    # Plot state trajectory
    ax.scatter(
        timesteps,
        state_trajectory[:, 0],
        state_trajectory[:, 1],
        c=state_trajectory[:, 1],
        cmap="viridis",
    )

    plt.savefig("state_trajectory_3d.png")


# Initialize environment, SAC agent, and replay buffer
env = OptimalControlEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = SAC(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

# Training loop
num_episodes = 200
max_timesteps = 500
batch_size = 64


final_rewards, state_trajectory, control_input_history = train(
    env, num_episodes, max_timesteps, batch_size
)

plot_rewards(final_rewards)
plot_state_trajectory(state_trajectory)


# TODO: get the u(t) for the last episode?
# Print final control input
final_control_input = control_input_history[-1]
print(f"Final control input: {final_control_input}")


# Close the environment
env.close()
