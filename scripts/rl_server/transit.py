from typing import Union
from multiprocessing import Manager
import random
import os


class Transition:
    def __init__(self, state, action, reward, next_state, value=0):
        self.state = state
        self.action = action
        self.reward = reward
        self.value = value
        self.next_state = next_state
        self.is_end = True if next_state is None else False
        self.is_end_val = 0 if next_state is None else 1

    def __repr__(self):
        return str(self.state) + ' with ' + str(self.action) + ' to ' + str(self.next_state) + ' r: ' + str(self.reward) + ' v: ' + str(self.value)


class StepData:
    def __init__(self):
        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None
        self.done = False

    def __repr__(self):
        return f'{self.state}, {self.next_state}, {self.reward}, {self.action}, {self.done}'




class SharedBuffer:
    def __init__(self):
        self.min_size = 1000
        self.size = 100000
        self.list: list[Union[StepData, None]] = Manager().list([None for _ in range(self.size)])
        self.index = Manager().Value('i', 0)
        self.max_index = Manager().Value('i', 0)
        self.step = Manager().Value('i', 0)
        self.lock = Manager().Lock()

    def add(self, data):
        with self.lock:
            self.step.value += 1
            index = self.index.value
            if index > self.max_index.value:
                self.max_index.value = index
            self.index.value += 1
            self.index.value = self.index.value % self.size
        self.list[index] = data

    def get_rand(self, request_number):
        res = []
        if self.max_index.value < self.min_size:
            return res
        number = min(request_number, self.max_index.value)
        ran = [random.randint(0, self.max_index.value) for _ in range(number)]
        for r in ran:
            res.append(self.list[r])
        return res

    def __str__(self):
        return f'SharedBuffer {self.index} {self.max_index} {self.step}'

    def save_to_file(self, path):
        all_t = []
        for i in range(self.max_index.value):
            d = self.list[i]
            t = f'{d.state}|{d.next_state}|{d.reward}|{d.action}|{d.is_end}'
            all_t.append(t)
        joint = '\n'.join(all_t)
        f = open(os.path.join(path, f'buffer_{self.step}'), 'w')
        f.write(joint)
        f.close()