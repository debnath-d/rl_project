import random
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def select_action(env, model, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_tensor)
        return action.squeeze(0).detach().numpy()


def train_dqn(model, memory, optimizer, batch_size, gamma=0.99):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.FloatTensor(np.array(actions))
    rewards = torch.FloatTensor(np.array(rewards))
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states)
    q_values_for_actions = (q_values * actions).sum(dim=-1, keepdim=True)
    next_q_values = model(next_states).detach()
    next_q_values_for_actions = (next_q_values * actions).sum(dim=-1, keepdim=True)

    target_q_values = rewards.unsqueeze(1) + gamma * next_q_values_for_actions * (
        1 - dones.unsqueeze(1)
    )

    loss = nn.MSELoss()(q_values_for_actions, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Return loss for logging
