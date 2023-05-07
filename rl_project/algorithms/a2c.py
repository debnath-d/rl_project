import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)


def train(
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    gamma=0.99,
    num_episodes=500,
    max_episode_steps=300,
    log_dir="runs/a2c",  # Add this argument for specifying the log directory
    device="cpu",  # Add this argument to accept the device
):
    writer = SummaryWriter(log_dir)  # Initialize the TensorBoard writer

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(max_episode_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            advantage = (
                reward
                + (1 - done) * gamma * critic(next_state_tensor)
                - critic(state_tensor)
            )
            critic_loss = advantage.pow(2).mean()

            actor_loss = (-actor(state_tensor) * advantage.detach()).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}: Total reward: {total_reward}")

        # Log metrics to TensorBoard
        writer.add_scalar("Total reward", total_reward, episode)
        writer.add_scalar("Actor loss", actor_loss.item(), episode)
        writer.add_scalar("Critic loss", critic_loss.item(), episode)

    writer.close()  # Close the TensorBoard writer
