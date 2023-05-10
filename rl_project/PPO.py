import torch
import torch.nn as nn
import torch.optim as optim

# import numpy as np
from torch.distributions import Normal
from env import OptimalControlEnv
import matplotlib.pyplot as plt

"""
    The Memory class is used to store the experiences of the agent in a memory buffer. The class defines the following methods:

    __init__: Initializes the memory buffers for states, actions, log-probabilities, rewards, and done flags.
    store: Stores a single experience tuple in the memory buffer.
    clear: Clears the memory buffers.
    The store method takes in five arguments: the current state of the environment, the action taken by the agent, 
    the log-probability of the action given the state, the reward received by the agent for taking the action, and 
    a flag indicating whether the episode has ended. The method stores each of these values in its corresponding memory buffer.

    The clear method simply resets all memory buffers to empty lists."""


class Memory:
    def __init__(self):
        # Initialize the memory buffers for states, actions, log-probabilities, rewards, and done flags
        self.states = []  # list to store the states
        self.actions = []  # list to store the actions
        self.logprobs = []  # list to store the log probabilities
        self.rewards = []  # list to store the rewards
        self.dones = []  # list to store the done flags

    def store(self, state, action, logprob, reward, done):
        """
        Store a single experience tuple in the memory buffer.

        :param state: the current state of the environment
        :param action: the action taken by the agent
        :param logprob: the log-probability of the action given the state
        :param reward: the reward received by the agent for taking the action
        :param done: a flag indicating whether the episode has ended
        """
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """
        Clear the memory buffers.
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []


"""The Actor and Critic classes are neural network models used to approximate the actor and critic functions, respectively. 
Both classes inherit from PyTorch's nn.Module class, and define the following methods:

__init__: Initializes the neural network architecture.
forward: Performs a forward pass through the network.
The Actor class takes in two arguments: the dimensionality of the state and action spaces. 
The neural network architecture consists of three fully connected layers: an input layer, a hidden layer with 64 units and
 a ReLU activation function, and an output layer with the same number of units as the action space. The forward method takes
in the current state and returns the action to take based on the current state.

The Critic class takes in one argument: the dimensionality of the state space. The neural network architecture consists of 
three fully connected layers: an input layer, a hidden layer with 64 units and a ReLU activation function, and an output 
layer with a single scalar value. The forward method takes in the current state and returns the estimated value of the state."""


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Initialize the actor network
        super(Actor, self).__init__()
        # Define the neural network architecture for the actor
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),  # input layer
            nn.ReLU(),  # activation function
            nn.Linear(64, 32),  # hidden layer
            nn.ReLU(),  # activation function
            nn.Linear(32, action_dim),  # output layer
        )

        """
        Perform a forward pass through the actor network.
        
        :param state: the current state of the environment
        :return: the action to take based on the current state
        """

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    # Initialize the critic network
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # Define the neural network architecture for the critic
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),  # input layer
            nn.ReLU(),  # activation function
            nn.Linear(64, 32),  # hidden layer
            nn.ReLU(),  # activation function
            nn.Linear(32, 1),  # output layer (scalar value)
        )

    """
        Perform a forward pass through the critic network.
        
        :param state: the current state of the environment
        :return: the estimated value of the state
        """

    def forward(self, state):
        return self.model(state)


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        action_std,
    ):
        """
        Initialize the PPO algorithm with the necessary hyperparameters

        :param state_dim: the dimensionality of the state space
        :param action_dim: the dimensionality of the action space
        :param lr_actor: the learning rate for the actor optimizer
        :param lr_critic: the learning rate for the critic optimizer
        :param gamma: the discount factor for future rewards
        :param K_epochs: the number of updates to perform on the network
        :param eps_clip: the clipping parameter for the ratio of new and old policy probabilities
        :param action_std: the standard deviation of the action distribution
        """
        # Create the actor and critic networks, and their optimizers
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # Store the hyperparameters
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std
        # Learning rate schedulers for the actor and critic optimizers
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=100, gamma=0.95
        )
        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=100, gamma=0.95
        )

    def select_action(self, state):
        """
        Select an action based on the current state, using the actor network and action distribution.

        :param state: the current state of the environment
        :return: the action to take and the log probability of the action
        """
        # Convert the state to a tensor and compute the action distribution
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        dist = Normal(action_probs, torch.tensor(self.action_std))
        # Sample an action and compute its log probability
        action = dist.sample()
        action_logprobs = dist.log_prob(action)

        return action.detach().numpy(), action_logprobs.detach()

    def normalize(self, x):
        """
        Normalize a tensor by subtracting its mean and dividing by its standard deviation.

        :param x: the tensor to normalize
        :return: the normalized tensor
        """
        x = x - x.mean()
        x = x / x.std()
        return x

    def update(self, memory):
        # Calculate the discounted rewards for each time step in the episode
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

            # Normalize the rewards

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Get the old states, actions, and log probabilities from the memory buffer

        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        # Normalize old_states
        old_states = self.normalize(old_states)

        # Loop for K_epochs to update the actor and critic networks
        for _ in range(self.K_epochs):
            # Compute the new policy probabilities and log probabilities
            logprobs = Normal(self.actor(old_states), torch.tensor(1.0)).log_prob(
                old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - self.critic(old_states).squeeze()
            # Normalize advantages
            # Normalize the advantages
            advantages = self.normalize(advantages)
            # Calculate the advantages by subtracting the estimated state values from the normalized rewards
            # Calculate the two surrogate losses
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            # Calculate the actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic(old_states).squeeze(), rewards)
            # Zero out the gradients, backpropagate the losses, and update the actor and critic networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        # Clear the memory buffer after updating the networks
        memory.clear()


# Create an empty list to store the total rewards for each episode
episode_rewards = []


def main():
    # create the environment and extract state and action dimensions
    env = OptimalControlEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # initialize the PPO agent with the state and action dimensions, learning rates, and other hyperparameters
    ppo = PPO(
        state_dim,
        action_dim,
        lr_actor=0.005,
        lr_critic=0.005,
        gamma=0.95,
        K_epochs=40,
        eps_clip=0.1,
        action_std=1.0,
    )
    # set the number of episodes to run and the maximum episode length
    num_episodes = 1000
    max_episode_length = 300
    # create a memory buffer to store the agent's experience during an episode
    memory = Memory()
    # loop through each episode
    for episode in range(num_episodes):
        # reset the environment and set the initial state
        state = env.reset()
        done = False
        episode_reward = 0
        # loop through each time step in the episode
        for step in range(max_episode_length):
            # select an action based on the current state
            action, logprob = ppo.select_action(state)
            # take a step in the environment based on the selected action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            # store the agent's experience in the memory buffer
            memory.store(state, action, logprob, reward, done)
            # update the current state and episode reward
            state = next_state
            episode_reward += reward
            # if the episode is done, exit the lood
            if done:
                break
        # update the agent's policy and value function based on the experiences in the memory buffer
        ppo.update(memory)
        ppo.actor_scheduler.step()
        # adjust the learning rate using a scheduler
        ppo.critic_scheduler.step()
        # record the episode reward
        episode_rewards.append(episode_reward)
        # print the episode number and reward
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    # plot the episode rewards
    plt.plot(episode_rewards)

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Rewards vs Episodes")

    plt.savefig("rewards.png")
    plt.figure()


if __name__ == "__main__":
    main()
    plt.figure()

    plt.plot(episode_rewards)

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Rewards vs Episodes")

    plt.savefig("rewards.png")
