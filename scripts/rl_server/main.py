from typing import Union
import numpy as np
from redis_server import RedisServer
from ddpg import DeepAC
import os
import signal
import datetime
from multiprocessing import Process, Pipe, Queue, connection
from multiprocessing.connection import Connection
from transit import Transition, StepData, SharedBuffer
from multiprocessing import Manager
import traceback
import sys
from logger import get_logger
import logging
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf.config.experimental.set_visible_devices([], 'GPU')

patch_number_max = 400
train_episode_number_max = 100
test_episode_number_max = 20
obs_size = 12
done = Manager().Value('i', 0)
run_name = '1'
trainer_count = 1
db_start = 1
train_embedded = True
get_model = False
train_random_action = True
test_random_action = True
decreased_random = True
best_action_percent = 0.9
generate_random = True
actor_layers = None
critic_layers = None
actor_optimizer = 'sgd'
critic_optimizer = 'sgd'
logger = get_logger()
logger.setLevel(level=logging.DEBUG)


def to_bool(val: str):
    if val.lower() in ['t', '1', 'true', 'y', 'yes']:
        return True
    return False


def to_list(val: str):
    res = []
    for layer in val.strip().split(','):
        res.append(float(layer))
    return res


args = sys.argv
if len(args) > 1:
    i = 1
    while i < len(args):
        arg = args[i]
        val = args[i + 1]
        if arg == 'run_name':
            run_name = val
        if arg == 'trainer_count':
            trainer_count = int(val)
        if arg == 'db_start':
            db_start = int(val)
        if arg == 'train_embedded':
            train_embedded = to_bool(val)
        if arg == 'get_model':
            get_model = to_bool(val)
        if arg == 'train_random_action':
            train_random_action = to_bool(val)
        if arg == 'test_random_action':
            test_random_action = to_bool(val)
        if arg == 'decreased_random':
            decreased_random = to_bool(val)
        if arg == 'best_action_percent':
            best_action_percent = float(val)
        if arg == 'generate_random':
            generate_random = to_bool(val)
        if arg == 'actor_layers':
            actor_layers = to_list(val)
        if arg == 'critic_layers':
            critic_layers = to_list(val)
        if arg == 'actor_optimizer':
            actor_optimizer = val
        if arg == 'critic_optimizer':
            critic_optimizer = val
        i += 2


def handler(signum, frame):
    global done
    done.value = 1


signal.signal(signal.SIGINT, handler)


def decode_obs(obs):
    res = f'''self.pos.r:{obs[0] * 150.0}, self.pos.th:{obs[1] * 180.0}, self.body.degree:{obs[2] * 180.0}, self.pos.x:{obs[3] * 52.5}, self.pos.y:{obs[4] * 34.0}, ball.pos.r:{obs[5] * 150.0}, ball.pos.th:{obs[6] * 180.0}, ball.pos.x:{obs[7] * 52.5}, ball.pos.y:{obs[8] * 34.0}, (ball.pos - self.pos).th:{obs[9] * 180.5}, ball.pos.dist(self.pos)){obs[10] * 150.0}, (ball.pos - self.pos).th() - self.body{obs[11] * 180.0}'''
    return res


def decode_act(act):
    res = f'''{act[0] * 180.0}'''
    return res


class PythonRLTrainer:
    def __init__(self,
                 shared_buffer: SharedBuffer,
                 model_receiver_q: Queue,
                 db_number: int,
                 start_time: str,
                 train_embedded: bool,
                 get_model: bool):
        self.shared_buffer: SharedBuffer = shared_buffer
        self.model_receiver_q: Queue = model_receiver_q
        self.train_embedded = train_embedded
        self.get_model = get_model
        self.player_count = 1
        self.observation_size = obs_size
        self.action_size = 1
        self.rl = DeepAC(observation_size=self.observation_size, action_size=self.action_size, shared_buffer=self.shared_buffer)
        self.rl.create_model_actor_critic(actor_layers=actor_layers, critic_layers=critic_layers, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)
        self.rd = RedisServer(db_number)
        self.rd.client.flushdb()
        self.training_episode_com_rewards = []
        self.testing_episode_com_rewards = []
        self.training_episode_res = []
        self.testing_episode_res = []
        self.latest_episode_rewards = []
        self.data: dict[int, StepData] = {}
        self.cycle = -1
        self.out_path = os.path.join('res', run_name + '_' + start_time, str(db_number))
        os.makedirs(self.out_path, exist_ok=True)
        self.init_trainer()
        self.patch_number = 0
        self.train_episode_number = 0
        self.test_episode_number = 0
        self.is_train = True

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
            self.shared_buffer.add(Transition(self.data[key].state, self.data[key].action, self.data[key].reward, self.data[key].next_state))
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
        global done
        logger.info('Wait for trainer')
        key, msg = self.rd.get_msg_from(num=0, cycle=None, done=done)
        logger.info(f'Receive from trainer: {key}, {msg}')
        if done.value == 1:
            return
        self.cycle = msg + 1
        if not isinstance(msg, int):
            raise Exception('the start message from trainer should be int.')
        self.rd.set_msg(key, 'OK')
        logger.info(f'Send OK to trainer')

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
            self.rl.update_actor(received_actor)

    def run_trainer_one_step(self):
        pre_num_cycle, values = self.rd.get_msg_from(num=0, msg_length=[2], cycle=self.cycle, wait_time_second=1, done=done)
        logger.info(f'trainer received({self.cycle}): {pre_num_cycle}, {values}')
        if pre_num_cycle is None:
            logger.info(f'trainer sent ' + RedisServer.FROM_AGENT_PRE_POSE + '_' + str(0) + '_' + str(self.cycle) + ', OK')
            self.rd.set_msg(RedisServer.FROM_AGENT_PRE_POSE + '_' + str(0) + '_' + str(self.cycle), 'OK')
        else:
            self.cycle = int(pre_num_cycle.split('_')[-1])
            is_start, is_done, status, reward = self.add_trainer_info(pre_num_cycle, values, self.is_train)
            logger.info(f'cycle:{self.cycle} is_start:{is_start} is_done:{is_done} status:{status} reward:{reward}')
            if not is_start:
                self.latest_episode_rewards.append(reward)
            logger.info(f'trainer sent ' + f'{pre_num_cycle}' + ', OK')
            self.rd.set_msg(pre_num_cycle, 'OK')
            if is_start:  # start
                self.rl.random_process.reset_states()
            elif is_done:  # end
                if self.is_train:
                    self.training_episode_res.append(status)
                    self.training_episode_com_rewards.append(sum(self.latest_episode_rewards))
                    self.latest_episode_rewards = []
                    self.train_episode_number += 1
                    if self.train_episode_number == train_episode_number_max:
                        self.is_train = False
                        self.train_episode_number = 0
                else:
                    self.testing_episode_res.append(status)
                    self.testing_episode_com_rewards.append(sum(self.latest_episode_rewards))
                    self.latest_episode_rewards = []
                    self.test_episode_number += 1
                    if self.test_episode_number == test_episode_number_max:
                        self.is_train = True
                        self.test_episode_number = 0
                        self.patch_number += 1
                if self.patch_number == patch_number_max:
                    self.end_function()
                    return False
        return True

    def run_player_one_step(self):
        pre_num_cycle, msg = self.rd.get_msg_from(num=1, msg_length=[obs_size], cycle=self.cycle, wait_time_second=0.5, done=done)
        logger.warning(f'player received({self.cycle}): {pre_num_cycle}, {msg}')
        if pre_num_cycle is not None:
            if isinstance(msg, str):  # FAKE message
                logger.warning(f'player sent ' + f'{pre_num_cycle}' + ', OK' + ' (Fake MSG)')
                self.rd.set_msg(pre_num_cycle, "OK")
            else:
                logger.warning(decode_obs(msg))
                if self.is_train:
                    self.add_player_info(pre_num_cycle, msg)
                random_percentage = 0.0
                if self.is_train and train_random_action:
                    if decreased_random:
                        random_percentage = None
                    else:
                        random_percentage = 1 - best_action_percent
                if not self.is_train and test_random_action:
                    random_percentage = 1 - best_action_percent

                action_arr = self.rl.get_random_action(msg, self.patch_number, patch_number_max, random_percentage, generate_random)
                logger.warning(f'q: {self.rl.get_q(msg, action_arr)}')
                # for a in range(-18, 18):
                #     ac = np.array([a * 10.0 / 180.0])
                #     ac = ac.reshape((1,))
                #     logger.warning(f'a:{a * 10.0}q: {self.rl.get_q(msg, ac)}')
                action_tmp = action_arr.tolist()
                action = []
                for a in action_tmp:
                    action.append(float(a))
                logger.warning(f'player selected action: {action_arr}:{decode_act(action)}')
                logger.warning(f'player sent {pre_num_cycle}, {action}')
                self.rd.set_msg(pre_num_cycle, action)
                if self.is_train:
                    self.add_player_action(pre_num_cycle, action)
        else:
            logger.warning(f'player sent ' + ' response old msg')
            self.response_old_message()

    def run(self):
        logger.critical('Start')
        while True:
            if self.get_model:
                self.get_and_update_actor()
            if done.value == 1 or self.patch_number % 50 == 0:
                self.end_function()
            if done.value == 1:
                break

            if not self.run_trainer_one_step():
                break

            self.run_player_one_step()

            if self.is_train:
                self.add_data_to_buffer(self.cycle)
                if self.train_embedded:
                    self.rl.update(32)
            self.cycle += 1
        logger.critical('End')


def run_manager(shared_buffer: SharedBuffer, model_receiver_q: Queue, db, start_time, train_embedded, get_model):
    python_rl_trainer = PythonRLTrainer(shared_buffer, model_receiver_q, db, start_time, train_embedded, get_model)
    python_rl_trainer.run()


def run_model(shared_buffer: SharedBuffer, all_model_sender_q: list[Queue], start_time):
    try:
        out_path = os.path.join('res', run_name + '_' + start_time, str(0))
        os.makedirs(out_path, exist_ok=True)
        out_file = open(os.path.join(out_path, 'log'), 'w')
        rl = DeepAC(observation_size=obs_size, action_size=1, train_interval_step=1, target_update_interval_step=200, shared_buffer=shared_buffer)
        rl.create_model_actor_critic(actor_layers=actor_layers, critic_layers=critic_layers, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)
        for model_sender_q in all_model_sender_q:
            model_sender_q.put(rl.actor.get_weights())
        last_online_update_step = 0
        last_send_model_step = 0
        step_in_each_update = 32
        online_update_step_interval = 1
        send_model_step_interval = 200
        while True:
            if done.value == 1:
                break
            step = shared_buffer.step.value
            if step <= shared_buffer.min_size:
                continue
            if step - last_online_update_step < online_update_step_interval:
                continue
            update_count = int((step - last_online_update_step) / online_update_step_interval)
            if update_count == 0:
                continue
            send_model = False
            if step - last_send_model_step >= send_model_step_interval:
                send_model = True
                last_send_model_step = step
            last_online_update_step += update_count * online_update_step_interval
            data_in_update = min(320, step_in_each_update * update_count)
            out_file.write(f'#{step} update {last_online_update_step} {data_in_update}\n')
            rl.update(data_in_update)
            if send_model:
                out_file.write(f'#{step} sent model\n')
                for model_sender_q in all_model_sender_q:
                    model_sender_q.put(rl.actor.get_weights())
            if 1000 < step < 2000:
                out_file.flush()
        print('done main model')
    except Exception as e:
        print(traceback.format_exc())


start_time = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

shared_buffer = SharedBuffer()
queues = [Queue() for i in range(trainer_count)]  # to get actor
ps = []
for i in range(trainer_count):
    p = Process(target=run_manager, args=(shared_buffer, queues[i], i + db_start, start_time, train_embedded, get_model))
    p.start()
    ps.append(p)
if not train_embedded:
    m = Process(target=run_model, args=(shared_buffer, queues, start_time))
    m.start()
for p in ps:
    p.join()
    print('end')
done.value = True
if not train_embedded:
    m.join()

