from redis_server import RedisServer
from ddpg import DeepAC
import os
import signal
import datetime
from multiprocessing import Process, Pipe, Queue, connection
from multiprocessing.connection import Connection
from transit import Transition


patch_number_max = 1000
train_episode_number_max = 100
test_episode_number_max = 10
done = False


def handler(signum, frame):
    global done
    done = True


signal.signal(signal.SIGINT, handler)


class StepData:
    def __init__(self):
        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None
        self.done = False


class PythonRLTrainer:
    def __init__(self, buffer_q: Queue, model_receiver_q: Queue, db_number, start_time):
        self.buffer_q: Queue = buffer_q
        self.model_receiver_q: Queue = model_receiver_q
        self.player_count = 1
        self.observation_size = 6
        self.action_size = 1
        self.rl = DeepAC(observation_size=self.observation_size, action_size=self.action_size)
        self.rl.create_model_actor_critic()
        self.rd = RedisServer(db_number)
        self.rd.client.flushdb()
        self.training_episode_com_rewards = []
        self.testing_episode_com_rewards = []
        self.training_episode_res = []
        self.testing_episode_res = []
        self.latest_episode_rewards = []
        self.data: dict[int, StepData] = {}
        self.cycle = -1
        self.out_path = os.path.join('res', 'name_' + start_time, str(db_number))
        os.makedirs(self.out_path, exist_ok=True)
        self.init_trainer()

    def add_trainer_info(self, pre_num_cycle, values, is_train):
        cycle = int(pre_num_cycle.split('_')[-1])
        is_done = int(values[0]) >= 2
        is_start = int(values[0]) == 0
        status = int(values[0])
        if is_start:
            return is_start, is_done, status, 0
        if is_train:
            reward_cycle = cycle - 1
            if reward_cycle not in self.data.keys():
                self.data[reward_cycle] = StepData()
            self.data[reward_cycle].done = is_done
            self.data[reward_cycle].reward = values[1]
            if is_done:
                self.data[reward_cycle].next_state = None
        return is_start, is_done, status, values[1]

    def add_player_info(self, pre_num_cycle, values):
        cycle = int(pre_num_cycle.split('_')[-1])
        if cycle not in self.data.keys():
            self.data[cycle] = StepData()
        self.data[cycle].state = values
        cycle -= 1
        if cycle not in self.data.keys():
            self.data[cycle] = StepData()
        if self.data[cycle].done is False:
            self.data[cycle].next_state = values

    def add_player_action(self, pre_num_cycle, action):
        cycle = int(pre_num_cycle.split('_')[-1])
        if cycle not in self.data.keys():
            self.data[cycle] = StepData()
        self.data[cycle].action = action

    def add_data_to_buffer(self, current_cycle):
        should_remove = []
        for key in self.data.keys():
            if self.data[key].next_state is None and self.data[key].done is False:
                if key < current_cycle - 2:
                    should_remove.append(key)
                continue
            if self.data[key].reward is None:
                if key < current_cycle - 2:
                    should_remove.append(key)
                continue
            if self.data[key].state is None:
                if key < current_cycle - 2:
                    should_remove.append(key)
                continue
            if self.data[key].action is None:
                if key < current_cycle - 2:
                    should_remove.append(key)
                continue
            self.buffer_q.put(Transition(self.data[key].state, self.data[key].action, self.data[key].reward, self.data[key].next_state))
            # self.rl.add_to_buffer(Transition(self.data[key].state, self.data[key].action, self.data[key].reward, self.data[key].next_state))
            should_remove.append(key)
        for key in should_remove:
            del self.data[key]

    def end_function(self):
        f = open(os.path.join(self.out_path, 'training_episode_com_rewards'), 'w')
        f.write('\n'.join([str(i) for i in self.training_episode_com_rewards]))
        f = open(os.path.join(self.out_path, 'testing_episode_com_rewards'), 'w')
        f.write('\n'.join([str(i) for i in self.testing_episode_com_rewards]))
        f = open(os.path.join(self.out_path, 'training_episode_res'), 'w')
        f.write('\n'.join([str(i) for i in self.training_episode_res]))
        f = open(os.path.join(self.out_path, 'testing_episode_res'), 'w')
        f.write('\n'.join([str(i) for i in self.testing_episode_res]))
        self.rl.save_weight(self.out_path)

    def wait_for_trainer(self):
        key, msg = self.rd.get_msg_from(num=0, cycle=None)
        self.cycle = msg + 1
        if not isinstance(msg, int):
            raise Exception('the start message from trainer should be int.')
        self.rd.set_msg(key, 'OK')

    def init_trainer(self):
        self.wait_for_trainer()

    def response_old_message(self):
        keys: list = self.rd.client.keys()
        for k in keys:
            key = k.decode()
            if not key.startswith(self.rd.FROM_AGENT_PRE_POSE):
                continue
            key_split = str(key).split('_')
            if len(key_split) != 3:
                continue

            num = int(key_split[1])
            cycle = int(key_split[2])
            if num > 0 and cycle < self.cycle:
                self.rd.set_msg(key, 'OK')
                self.rd.client.delete(key)

    def get_and_update_actor(self):
        received_actor = None
        while True:
            if self.model_receiver_q.empty():
                break
            received_actor = self.model_receiver_q.get()
        if received_actor is not None:
            print('h1')
            self.rl.update_actor(received_actor)
            print('h2')

    def run(self):
        patch_number = 0
        train_episode_number = 0
        test_episode_number = 0
        is_train = True

        while True:
            self.get_and_update_actor()
            if done or patch_number % 50 == 0:
                self.end_function()
            if done:
                return
            pre_num_cycle, values = self.rd.get_msg_from(num=0, msg_length=[2], cycle=self.cycle, wait_time_second=1)
            if pre_num_cycle is None:
                self.rd.set_msg(RedisServer.FROM_AGENT_PRE_POSE + '_' + str(0) + '_' + str(self.cycle), 'OK')
            if pre_num_cycle is not None:
                self.cycle = int(pre_num_cycle.split('_')[-1])
                is_start, is_done, status, reward = self.add_trainer_info(pre_num_cycle, values, is_train)
                if not is_start:
                    self.latest_episode_rewards.append(reward)
                self.rd.set_msg(pre_num_cycle, 'OK')
                if is_start:  # start
                    pass
                elif is_done:  # end
                    if is_train:
                        self.training_episode_res.append(status)
                        self.training_episode_com_rewards.append(sum(self.latest_episode_rewards))
                        self.latest_episode_rewards = []
                        train_episode_number += 1
                        if train_episode_number == train_episode_number_max:
                            is_train = False
                            train_episode_number = 0
                    else:
                        self.testing_episode_res.append(status)
                        self.testing_episode_com_rewards.append(sum(self.latest_episode_rewards))
                        self.latest_episode_rewards = []
                        test_episode_number += 1
                        if test_episode_number == test_episode_number_max:
                            is_train = True
                            test_episode_number = 0
                            patch_number += 1
                    if patch_number == patch_number_max:
                        self.end_function()
                        return

            pre_num_cycle, msg = self.rd.get_msg_from(num=1, msg_length=[6], cycle=self.cycle, wait_time_second=0.5)
            if pre_num_cycle is not None:
                if isinstance(msg, str):  # FAKE message
                    self.rd.set_msg(pre_num_cycle, "OK")
                else:
                    if is_train:
                        self.add_player_info(pre_num_cycle, msg)
                    action_arr = self.rl.get_random_action(msg, patch_number, patch_number_max, None if is_train else 0.0)
                    action_tmp = action_arr.tolist()
                    action = []
                    for a in action_tmp:
                        action.append(float(a))
                    self.rd.set_msg(pre_num_cycle, action)
                    if is_train:
                        self.add_player_action(pre_num_cycle, action)
            else:
                self.response_old_message()
            if is_train:
                self.add_data_to_buffer(self.cycle)
            self.cycle += 1


def run_manager(buffer_q: Queue, model_receiver_q: Queue, db, start_time):
    python_rl_trainer = PythonRLTrainer(buffer_q, model_receiver_q, db, start_time)
    python_rl_trainer.run()


def run_model(buffer_q: Queue, all_model_sender_q: list[Queue]):
    rl = DeepAC(observation_size=6, action_size=1, train_interval_step=1, target_update_interval_step=10)
    rl.create_model_actor_critic()
    for model_sender_q in all_model_sender_q:
        model_sender_q.put(rl.actor.get_weights())
    while True:
        i = 0
        while not buffer_q.empty():
            rl.add_to_buffer(buffer_q.get())
            i+=1
        while i > 0:
            rl.update()
            i -= 1
        if rl.step_number % 40 == 0:
            for model_sender_q in all_model_sender_q:
                model_sender_q.put(rl.actor.get_weights())

start_time = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
trainer_count = 1
queue = Queue()  # to update buffer
queues = [Queue() for i in range(trainer_count)]  # to get actor
ps = []
for i in range(trainer_count):
    p = Process(target=run_manager, args=(queue, queues[i], i + 1, start_time))
    p.start()
    ps.append(p)
m = Process(target=run_model, args=(queue, queues))
m.start()
for p in ps:
    p.join()
    print('end')
m.join()

