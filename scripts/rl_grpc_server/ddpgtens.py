import os
import tensorflow as tf
import numpy as np
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_dim=state_dim)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.dense1(tf.expand_dims(state, axis=0))
        x = self.dense2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_dim=state_dim + action_dim)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        action = tf.reshape(tf.cast(action, dtype=tf.float32), shape=(-1, 1))
        state = tf.cast(state, dtype=tf.float32)
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def select_action(self, state):
        return np.clip(self.actor(state).numpy(), -1.0, 1.0)

    def update(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state)
            target_q_values = self.target_critic(next_state, target_actions)
            y = reward + (1. - done) * 0.99 * target_q_values

            q_values = self.critic(state, action)
            critic_loss = tf.reduce_mean(tf.square(y - q_values))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            predicted_actions = self.actor(state)
            actor_loss = -tf.reduce_mean(self.critic(state, predicted_actions))

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.update_target_networks()

    def update_target_networks(self):
        tau = 0.005
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()

        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]

        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

class DeepAC:
    def __init__(self, state_dim=2, action_dim=1):
        self.agent = DDPG(state_dim, action_dim)
        self.reply_buffer = ReplayBuffer(buffer_size=10000)

    def get_random_action(self, state):
        action = np.clip(self.agent.select_action(state) + np.random.normal(0, 0.1), -1.0, 1.0)
        return action.tolist()

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.reply_buffer.push(state, action, reward, next_state, done)
        if len(self.reply_buffer.buffer) >= 64:
            states, actions, rewards, next_states, dones = self.reply_buffer.sample(64)
            self.agent.update(states, actions, rewards, next_states, dones)


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
# Example usage:
# game = Game()
# agent = DeepAC(2, 1)

# rewards_100 = 0
# for episode in range(10000):
#     state = game.reset()
#     sum_reward = 0
#     for step in range(100):
#         action = agent.get_random_action(state)
#         next_state, reward, done = game.step(action)
#         sum_reward += reward
#         agent.add_to_buffer(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             break
#     rewards_100 += sum_reward
#     if episode % 100 == 0:
#         print(episode, rewards_100 / 100)
#         rewards_100 = 0
