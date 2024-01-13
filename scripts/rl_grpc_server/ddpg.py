import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transit import Transition
from reply_buffer_simple import Buffer
import random
import random
import copy
import numpy as np


#Set Hyperparameters
# Hyperparameters adapted for performance from
#https://ai.stackexchange.com/questions/22945/ddpg-doesnt-converge-for-mountaincarcontinuous-v0-gym-environment
capacity=1000000
batch_size=64
update_iteration=1
tau=0.001 # tau for soft updating
gamma=0.99 # discount factor
directory = './'
hidden1=2 # hidden layer for actor
hidden2=10 #hiiden laye for critic
state_dim = 2
action_dim = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=capacity):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        state: np.array
            batch of state or observations
        action: np.array
            batch of actions executed given a state
        reward: np.array
            rewards received as results of executing action
        next_state: np.array
            next state next state or observations seen after executing action
        done: np.array
            done[i] = 1 if executing ation[i] resulted in
            the end of an episode and 0 otherwise.
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in ind:
            st, n_st, act, rew, dn = self.storage[i]
            state.append(np.array(st, copy=False))
            next_state.append(np.array(n_st, copy=False))
            action.append(np.array(act, copy=False))
            reward.append(np.array(rew, copy=False))
            done.append(np.array(dn, copy=False))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)

class ClipLayer(nn.Module):
    def __init__(self, min_value, max_value):
        super(ClipLayer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=self.min_value, max=self.max_value)
    
    
class Actor(nn.Module):
    """
    The Actor model takes in a state observation as input and 
    outputs an action, which is a continuous value.
    
    It consists of four fully connected linear layers with ReLU activation functions and 
    a final output layer selects one single optimized action for the state
    """
    def __init__(self, n_states, action_dim, hidden1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden1), 
            nn.ReLU(), 
            # nn.Linear(hidden1, hidden1), 
            # nn.ReLU(), 
            nn.Linear(hidden1, 1),
            nn.Tanh()
                        # nn.BatchNorm1d(hidden1),  # Add BatchNorm1d layer

            # ClipLayer(min_value=-0.1, max_value=0.1)
        )

    def forward(self, state):
        o = self.net(state)
        # o = torch.clamp(o, -0.1, 0.1)  # Clip the output between -0.1 and 0.1
        return o

class Critic(nn.Module):
    """
    The Critic model takes in both a state observation and an action as input and 
    outputs a Q-value, which estimates the expected total reward for the current state-action pair. 
    
    It consists of four linear layers with ReLU activation functions, 
    State and action inputs are concatenated before being fed into the first linear layer. 
    
    The output layer has a single output, representing the Q-value
    """
    def __init__(self, n_states, action_dim, hidden2):
        super(Critic, self).__init__()
        print(n_states, action_dim, hidden2)
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, hidden2), 
            nn.ReLU(), 
            # nn.Linear(hidden2, hidden2), 
            # nn.ReLU(), 
            nn.Linear(hidden2, hidden2), 
            nn.ReLU(), 
            nn.Linear(hidden2, action_dim),
            # nn.Sigmoid()
                        # nn.BatchNorm1d(hidden1),  # Add BatchNorm1d layer

        )
        
    def forward(self, state, action):
        state = state.reshape(64, 2)
        action = action.reshape(64, 1)
        return self.net(torch.cat((state, action), 1))

class OU_Noise(object):
    """Ornstein-Uhlenbeck process.
    code from :
    https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    The OU_Noise class has four attributes
    
        size: the size of the noise vector to be generated
        mu: the mean of the noise, set to 0 by default
        theta: the rate of mean reversion, controlling how quickly the noise returns to the mean
        sigma: the volatility of the noise, controlling the magnitude of fluctuations
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample.
        This method uses the current state of the noise and generates the next sample
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state



class DDPG(object):
    def __init__(self, state_dim, action_dim):
        """
        Initializes the DDPG agent. 
        Takes three arguments:
               state_dim which is the dimensionality of the state space, 
               action_dim which is the dimensionality of the action space, and 
               max_action which is the maximum value an action can take. 
        
        Creates a replay buffer, an actor-critic  networks and their corresponding target networks. 
        It also initializes the optimizer for both actor and critic networks alog with 
        counters to track the number of training iterations.
        """
        self.replay_buffer = Replay_buffer()
        
        self.actor = Actor(state_dim, action_dim, hidden1).to(device)
        self.actor_target = Actor(state_dim, action_dim,  hidden1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim,  hidden2).to(device)
        self.critic_target = Critic(state_dim, action_dim,  hidden2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)
        # learning rate

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.actor.apply(self.weights_init)
        self.critic.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)
            
    def select_action(self, state):
        """
        takes the current state as input and returns an action to take in that state. 
        It uses the actor network to map the state to an action.
        """
        
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    sum_current_Q = 0
    sum_q_error = 0
    count = 0
    sum_actor_loss = 0
    def update(self):
        """
        updates the actor and critic networks using a batch of samples from the replay buffer. 
        For each sample in the batch, it computes the target Q value using the target critic network and the target actor network. 
        It then computes the current Q value 
        using the critic network and the action taken by the actor network. 
        
        It computes the critic loss as the mean squared error between the target Q value and the current Q value, and 
        updates the critic network using gradient descent. 
        
        It then computes the actor loss as the negative mean Q value using the critic network and the actor network, and 
        updates the actor network using gradient ascent. 
        
        Finally, it updates the target networks using 
        soft updates, where a small fraction of the actor and critic network weights are transferred to their target counterparts. 
        This process is repeated for a fixed number of iterations.
        """

        for it in range(update_iteration):
            # For each Sample in replay buffer batch
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1-done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic.forward(state, action)
            #average of current_Q
            current_Q_avg = current_Q.mean()
            DDPG.sum_current_Q += current_Q_avg
            DDPG.count += 1
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            DDPG.sum_q_error += critic_loss.mean()
            
            
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -self.critic(state, self.actor(state)).mean() 
            DDPG.sum_actor_loss += actor_loss.mean()
            if DDPG.count % 1000 == 0:
                print(DDPG.sum_current_Q / 1000, DDPG.sum_q_error / 1000, DDPG.sum_actor_loss / 1000)
                DDPG.sum_current_Q = 0
                DDPG.count = 0            
                DDPG.sum_q_error = 0
                DDPG.sum_actor_loss = 0
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            
            """
            Update the frozen target models using 
            soft updates, where 
            tau,a small fraction of the actor and critic network weights are transferred to their target counterparts. 
            """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
           
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')

    def load(self):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        

class DeepAC:
    count = 0
    sum_action = 0
    sum_action2 = 0
    currect_action = 0
    def __init__(self) -> None:
        global state_dim, action_dim
        self.agent = DDPG(state_dim, action_dim)
        
    def GetRandomBestAction(self, state):
        DeepAC.count += 1
        action = self.agent.select_action(state) / 10.0
        DeepAC.sum_action += abs(action)
        DeepAC.sum_action2 += action
        if state[0] < state[1] and action > 0:
            DeepAC.currect_action += 1
        if state[0] > state[1] and action < 0:
            DeepAC.currect_action += 1
        if DeepAC.count % 1000 == 0:
            print(DeepAC.count, DeepAC.sum_action / 1000, DeepAC.sum_action2 / 1000, DeepAC.currect_action / 1000)
            DeepAC.sum_action = 0
            DeepAC.sum_action2 = 0
            DeepAC.currect_action = 0
        action = (action + np.random.normal(0, 0.01, size=action_dim)).clip(-0.05, 0.05)
        return action
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.agent.replay_buffer.push((state, next_state, action, reward, np.float32(np.array([done]))))
        self.agent.update()
 
 
class Game:
    def __init__(self):
        self.player_position = 0.0
        self.goal_position = 0.0
        self.episode_steps = 100
        self.step_count = 0

    def reset(self):
        self.player_position = 0.0
        self.goal_position = np.random.uniform(-1, 1)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return np.array([float(self.player_position), float(self.goal_position)])

    def step(self, action):
        self.step_count += 1
        self.player_position += action

        if self.player_position < -1 or self.player_position > 1:
            done = True
            reward = -2
        elif abs(self.player_position - self.goal_position) < 0.2:
            done = True
            reward = 1
        else:
            done = False
            reward = 0

        if self.step_count >= self.episode_steps:
            done = True

        if not done:
            reward += -0.01 if abs(self.player_position - self.goal_position) < abs(self.player_position + action - self.goal_position) else +0.01

        return self.get_state(), reward, done

game = Game()
agent = DeepAC()

rewards_100 = 0
for episode in range(10000):
    state = game.reset()
    sum_reward = 0
    for step in range(100):
        action = agent.GetRandomBestAction(state)
        next_state, reward, done = game.step(action)
        sum_reward += reward
        agent.add_to_buffer(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    rewards_100 += sum_reward
    if episode % 100 == 0:
        print(episode, rewards_100 / 100)
        rewards_100 = 0