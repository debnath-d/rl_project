import argparse

from algorithms.a2c import Actor, Critic, train
from env import OptimalControlEnv
import torch
from torch import optim


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
