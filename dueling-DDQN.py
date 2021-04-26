import random
import sys
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import matplotlib.pyplot as plt
import argparse

from utils import discrete_to_continuous

class DuelingDQNAgent:

    def __init__(self, state_size, action_size, seed, parameters):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.parameters = parameters
        
        self.qnetwork_stable = DuelingQNetwork(state_size, action_size, seed)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_stable.parameters(), lr=self.parameters.LR)

        # replay memory
        self.memory = deque(maxlen=self.parameters.BUFFER_SIZE)  
        self.batch_size = self.parameters.BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # timestep
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
       
        self.add_to_memory(state, action, reward, next_state, done)
        
        if self.t_step + 1 == self.parameters.UPDATE_EVERY:
            self.t_step = 0
        
        if self.t_step == 0:
            
            if len(self.memory) > self.parameters.BATCH_SIZE:
                self.learn(self.sample_from_memory(), self.parameters.GAMMA)

    
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_stable.eval()
        
        with torch.no_grad() as nograd:
            action_values = self.qnetwork_stable(state)

        self.qnetwork_stable.train()

        # exploration-explotation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
              
    def learn(self, raw_memory_experience, gamma):
        
        loss = self.compute_loss(raw_memory_experience, gamma)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # pdate target network
        self.soft_update(self.qnetwork_stable, self.qnetwork_target, self.parameters.TAU)  

    def compute_loss(self, experiences, gamma):
        states, actions, rewards, next_states, done_array = experiences

        Q_argmax = self.qnetwork_stable(next_states).detach()

        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_argmax.max(1)[1].unsqueeze(1))
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done_array))

        Q_expected = self.qnetwork_stable(states).gather(1, actions)

        # loss
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def soft_update(self, local_network, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))
    
    def sample_from_memory(self):
        # Randomly sample a batch of experiences from memory.
        experiences = random.sample(self.memory, k=self.batch_size)
        clean_exp = []
        for e in experiences:
            if e is not None:
                clean_exp.append(e)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        done_array = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
  
        return (states, actions, rewards, next_states, done_array)


def train(parameters):

    number_of_episodes = parameters.n_episodes
    max_t = parameters.max_t
    eps_start = parameters.eps_start
    eps_end = parameters.eps_end
    eps_decay = parameters.eps_decay

    env = gym.make('LunarLanderContinuous-v2')
    env.seed(0)

    shape_of_action = 72

    agent = DuelingDQNAgent(state_size=8, action_size=shape_of_action, seed=0, parameters=parameters)

    scores = []
    solve_iterations_num = 100
    score_memory = deque(maxlen=solve_iterations_num)
    epsilon = eps_start          
    for episode_iter in range(number_of_episodes):
        state, score = reset_episode(env)
        for t in range(max_t):
            action = agent.act(state, epsilon)
            continuousAction = discrete_to_continuous(action)
            next_state, reward, done, _ = env.step(continuousAction)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            # # adding noise
            # noise = np.random.normal(0,0.05,2)
            # state[0] = state[0] + noise[0]
            # state[1] = state[1] + noise[1]
            score += reward
            if done:
                break 
        score_memory.append(score)
        scores.append(score)
        # epsilon decay
        epsilon = max(eps_end, eps_decay*epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_iter, np.mean(score_memory)), end="")
        if episode_iter % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_iter, np.mean(score_memory)))

        if np.mean(score_memory)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_iter-solve_iterations_num, np.mean(score_memory)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_Dueling_DDQN.pth')
            break

    return scores

def reset_episode(env):
    state = env.reset()
    score = 0
    return state, score


class DuelingQNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, seed):
        super(DuelingQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = torch.manual_seed(seed)

        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(1).unsqueeze(1).expand(state.size(0), self.output_dim))
        
        return qvals


def parse_arguments():
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer-size', dest = 'BUFFER_SIZE',type=int, default=int(1e5)) # replay buffer size
    parser.add_argument('--batch-size', dest = 'BATCH_SIZE',type=int, default=int(64)) # batch size
    parser.add_argument('--gamma', dest = 'GAMMA',type=float, default=0.99) # discount factor
    parser.add_argument('--tau', dest = 'TAU',type=float, default=(1e-3)) # for soft update
    parser.add_argument('--lr', dest = 'LR',type=float, default=5e-4) # learning rate 
    parser.add_argument('--update-every', dest = 'UPDATE_EVERY',type=int, default=int(4)) # replay buffer size

    parser.add_argument('--episodes-number', dest = 'n_episodes',type=int, default=int(2000)) # number of episodes
    parser.add_argument('--max-t', dest = 'max_t',type=int, default=int(1000))
    parser.add_argument('--eps-start', dest = 'eps_start',type=float, default=1.0)
    parser.add_argument('--eps-end', dest = 'eps_end',type=float, default=0.01)
    parser.add_argument('--eps-decay', dest = 'eps_decay',type=float, default=0.995)

    return parser.parse_args()

def main(args):

    parameters = parse_arguments()

    scores = train(parameters)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("graph.png")

if __name__ == '__main__':

    main(sys.argv)