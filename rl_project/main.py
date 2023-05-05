from collections import deque

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_project.algorithms.dqn import DQN, select_action, train_dqn
from rl_project.env import OptimalControlEnv


def main():
    env = OptimalControlEnv()
    model = DQN(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = deque(maxlen=10000)

    num_episodes = 1000
    max_episode_length = 300
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    epsilon = epsilon_start

    # Create TensorBoard SummaryWriter
    writer = SummaryWriter()

    total_steps = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        for step in range(max_episode_length):
            if not episode % 10:
                print(f"{episode=}")
            action = select_action(env, model, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            memory.append((state, action, reward, next_state, done))
            loss = train_dqn(model, memory, optimizer, batch_size)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # Log metrics
            if loss is not None:
                writer.add_scalar("loss", loss, total_steps)
            writer.add_scalar("reward", reward, total_steps)

            if done:
                break

        # Log episode-level metrics
        writer.add_scalar("episode_reward", episode_reward, episode)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Close the SummaryWriter
    writer.close()


if __name__ == "__main__":
    main()
