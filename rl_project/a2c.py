import argparse

import torch
import torch.nn as nn
from env import OptimalControlEnv
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
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
        super().__init__()
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

    writer.flush()
    writer.close()  # Close the TensorBoard writer


def main(args):
    env = OptimalControlEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the models and move them to the device
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    # Pass the device to the train function
    train(
        env,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        args.gamma,
        args.num_episodes,
        args.max_episode_steps,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm", default="a2c", help="The algorithm to use (default: a2c)"
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=0.0003,
        help="The learning rate for the actor (default: 0.0003)",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=0.001,
        help="The learning rate for the critic (default: 0.001)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=500,
        help="Number of episodes to train (default: 500)",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)",
    )

    args = parser.parse_args()

    if args.algorithm.lower() == "a2c":
        main(args)
    else:
        print(f"Unsupported algorithm '{args.algorithm}'.")
