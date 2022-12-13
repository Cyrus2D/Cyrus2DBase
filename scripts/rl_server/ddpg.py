import os.path

from keras import models, layers, activations, optimizers, losses, metrics, regularizers
from keras.engine.sequential import Sequential
import random
from numpy import array
from typing import List
import keras
from keras import Model
import numpy as np
from reply_buffer_simple import Buffer
from transit import Transition
import keras.backend as K
import tensorflow as tf
# from main import patch_number_max
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        return tf.where(condition, squared_loss, linear_loss)  # condition, true, false


def clipped_error(y_true, y_pred):
    return K.mean(huber_loss(y_true, y_pred, np.inf), axis=-1)


def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min,
                                                       n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.size = size
        self.reset_states()
        self.x_prev = 0

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = np.random.normal(self.mu,self.current_sigma,self.size)


class DeepAC:
    def __init__(self, observation_size, action_size, buffer_size=100000, train_interval_step=1, target_update_interval_step=200):
        self.observation_size = observation_size
        self.action_size = action_size
        self.model_type = ''
        self.actor: Sequential = None
        self.target_actor: Sequential = None
        self.critic: Sequential = None
        self.target_critic: Sequential = None
        self.critic_action_input = None
        self.critic_action_input_idx = None
        self.actor_train_fn = None
        self.buffer = Buffer(buffer_size)
        self.train_interval_step = train_interval_step
        self.target_update_interval_step = target_update_interval_step
        self.action_number = 9
        self.transitions: List[Transition] = []
        self.gama = 0.95
        self.episode_number = 0
        self.plan_number = 0
        self.step_number = 0
        self.use_double = False
        self.loss_values = []
        self.target_model_update = 2.7#2.005
        self.random_process = OrnsteinUhlenbeckProcess(size=1, theta=.15, mu=0., sigma=.1)
        self.critic_history = []
        pass

    def create_model_actor_critic(self):
        self.model_type = 'param'
        input_obs = layers.Input((self.observation_size,))
        actor = layers.Dense(20, activation='relu')(input_obs)
        actor = layers.Dense(10, activation='relu')(actor)
        actor = layers.Dense(self.action_size, activation='tanh')(actor)
        actor = keras.Model(input_obs, actor)
        actor.summary()
        self.actor = actor

        input_obs = layers.Input((self.observation_size,), name='observation_input')
        flattened_observation = layers.Flatten()(input_obs)
        action_input = layers.Input(shape=(self.action_size,), name='action_input')
        critic = layers.Dense(20, activation='relu')(flattened_observation)
        critic = layers.Concatenate()([critic, action_input])
        critic = layers.Dense(20, activation='relu')(critic)
        critic = layers.Dense(10, activation='relu')(critic)
        critic = layers.Dense(1)(critic)
        critic = Model(inputs=[action_input, input_obs], outputs=critic)
        critic.summary()
        self.critic = critic
        self.critic_action_input = action_input
        self.critic_action_input_idx = self.critic.input.index(self.critic_action_input)

        self.target_actor = keras.models.clone_model(self.actor)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = keras.models.clone_model(self.critic)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # self.actor.compile(optimizer='sgd', loss='mse')

        # actor_optimizer = optimizers.RMSprop(lr=1e-4)
        actor_optimizer = optimizers.SGD(lr=0.001)
        # self.actor.compile(optimizer=actor_optimizer, loss='mse')
        # critic_optimizer = optimizers.RMSprop(lr=1e-4)
        critic_optimizer = optimizers.SGD(lr=0.001)
        if self.target_model_update < 1.:
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)

        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=['mae', 'mse'])

        combined_inputs = []
        state_inputs = []
        for i in self.critic.input:
            # print(i.name)
            # print(self.critic_action_input)
            if i.name == self.critic_action_input.name:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                state_inputs.append(i)

        combined_inputs[self.critic_action_input_idx] = self.actor(state_inputs)
        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(params=self.actor.trainable_weights, loss=-K.mean(combined_output))
        if self.target_model_update < 1.:
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates
        self.actor_train_fn = K.function(state_inputs + [K.learning_phase()], [self.actor(state_inputs)], updates=updates)

    def read_model(self, trained_actor_path, trained_critic_path):
        actor_read = keras.models.load_model(trained_actor_path)
        self.actor.set_weights(actor_read.get_weights())

    def read_weight(self, trained_actor_path, trained_critic_path):
        self.actor.load_weights(trained_actor_path)
        self.target_actor.load_weights(trained_actor_path)
        # self.critic.load_weights(trained_critic_path)
        # self.target_critic.load_weights(trained_critic_path)

    def save_weight(self, path):
        self.actor.save_weights(os.path.join(path, '_agent_actor_w.h5'))
        self.critic.save_weights(os.path.join(path, '_agent_critic_w.h5'))

    def pos_process(self, path):
        f = open(path + '_critic_history', 'w')
        f.write('loss,mae,mse\n')
        for h in self.critic_history:
            f.write(str(h['loss'][0]) + ',' + str(h['mae'][0]) + ',' + str(h['mse'][0]) + '\n')
        f.close()

    def get_best_action(self, state):
        batch = [state]
        batch = array(batch)
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (1,)
        return action

    def get_random_action(self, state, patch_number, patch_number_max, p_rnd=None):
        if p_rnd is None:
            max_number = patch_number_max
            number = patch_number
            max_number = max_number / 2
            if number > max_number:
                p_rnd = 0.1
            else:
                p_rnd = 1.0 + number / max_number * -0.9
        best_action = self.get_best_action(state)
        if random.random() < p_rnd:
            # noise = self.random_process.sample()
            # assert noise.shape == best_action.shape
            # best_action += noise
            best_action = np.array([random.random() * 2.0 - 1.0])
        return best_action

    def add_to_buffer(self, state, action, reward, next_state=None):
        self.buffer.add(Transition(state, action, reward, next_state))

        self.step_number += 1
        if self.step_number % self.train_interval_step == 0:
            self.update_from_buffer()
        if self.target_model_update > 1 and self.step_number % self.target_update_interval_step == 0:
            self.target_critic.set_weights(self.critic.get_weights())
        if self.target_model_update > 1 and self.step_number % self.target_update_interval_step == 0:
            self.target_actor.set_weights(self.actor.get_weights())
        if next_state is None:  # End step in episode
            self.episode_number += 1

    def update_from_buffer(self):
        transits: List[Transition] = self.buffer.get_rand(32)
        if len(transits) == 0:
            return
        is_image_param = False

        # print('buffer size:', self.buffer.i)
        states_view = []
        next_states_view = []
        states_param = []
        next_states_param = []
        terminal1_batch = []
        reward_batch = []
        action_batch = []
        for t in transits:
            reward_batch.append(t.reward)
            terminal1_batch.append(t.is_end_val)
            action_batch.append(t.action)
            states_view.append(t.state)
            if t.is_end:
                next_states_view.append(t.state)
            else:
                next_states_view.append(t.next_state)


        states_view = array(states_view)
        next_states_view = array(next_states_view)
        terminal1_batch = array(terminal1_batch)
        reward_batch = array(reward_batch)
        action_batch = array(action_batch)

        target_actions = self.target_actor.predict_on_batch(next_states_view)
        state1_batch_with_action = [next_states_view]
        state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
        target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
        discounted_reward_batch = self.gama * target_q_values
        discounted_reward_batch *= terminal1_batch
        targets = (reward_batch + discounted_reward_batch).reshape(32, 1)

        state0_batch_with_action = [states_view]
        state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
        history = self.critic.fit(x=state0_batch_with_action, y=targets, batch_size=32, verbose=0)
        self.critic_history.append(history.history)
        inputs = [states_view]
        action_values = self.actor_train_fn(inputs)[0]
