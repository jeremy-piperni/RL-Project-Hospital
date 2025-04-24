import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

def get_epsilon(episode, total_episodes, epsilon_start=1.0, epsilon_end=0.05):
    halfway = total_episodes // 2
    if episode < halfway:
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / halfway)
    else:
        epsilon = epsilon_end
    return epsilon

def train_DQN(env, episodes=500, batch_size=64, gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_end=0.05):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    epsilon = epsilon_start
    target_update_freq = 10

    total_rewards = []
    correct_diagnoses = 0
    success_rates = []
    examinations_per_diagnosis = []
    for episode in range(episodes):
        epsilon = get_epsilon(episode, episodes)
        state = env.reset()
        total_reward = 0
        done = False
        
        for t in range(env.max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        total_rewards.append(total_reward)
        if total_reward > -1:
            correct_diagnoses += 1
        examinations_per_diagnosis.append(t)

        success_rates.append(correct_diagnoses / (episode + 1))

    return policy_net, total_rewards, success_rates, examinations_per_diagnosis