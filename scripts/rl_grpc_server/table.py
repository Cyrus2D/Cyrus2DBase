import random
from typing import List
from reply_buffer_simple import Buffer
from transit import Transition


class QTable:
    def __init__(self, buffer_size=100000, train_interval_step=1):
        self.q_table = []
        for s1 in range(0, 37):
            self.q_table.append([])
            for a1 in range(0, 37):
                self.q_table[-1].append(0)
        self.buffer = Buffer(buffer_size)
        self.train_interval_step = train_interval_step
        self.transitions: List[Transition] = []
        self.gama = 0.95
        self.episode_number = 0
        self.plan_number = 0
        self.step_number = 0
        self.loss_values = []
        self.target_model_update = 2.005

    def discretize_state(self, state):
        s1 = int(state[0] * 360.0 / 10)
        return s1

    def discretize_action(self, action):
        a = int(action * 360.0 / 10.0)
        return a

    def continues_action(self, action):
        return action * 10.0 / 360.0

    def get_best_action_d(self, state_c):
        print(state_c)
        state_d = state_c[0]
        max_q = self.q_table[state_d][0]
        a_i = 0
        for i in range(36):
            if self.q_table[state_d][i] > max_q:
                max_q = self.q_table[state_d][i]
                a_i = i
        return a_i

    def GetRandomBestAction(self, state_c, epsilon):
        best_action_d = self.get_best_action_d(state_c)
        if random.random() < epsilon:
            best_action_d = random.randint(0, 35)
        return best_action_d

    def add_to_buffer(self, state, action, reward, next_state=None):
        self.buffer.add(Transition(state, action, reward, next_state))
        self.step_number += 1
        self.update_from_buffer()

    def update_from_buffer(self):
        transits: List[Transition] = self.buffer.get_rand(32)
        if len(transits) == 0:
            return
        for t in transits:
            d_state = t.state
            d_action = t.action
            max_q_s = 0
            if t.next_state is not None:
                next_state_best_action_d = self.get_best_action_d(t.next_state)
                d_next_state_d = t.next_state
                max_q_s = self.q_table[d_next_state_d[0]][next_state_best_action_d]
            self.q_table[d_state[0]][d_action] += 0.1 * (t.reward + 0.95 * (max_q_s - self.q_table[d_state[0]][d_action]))

