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
import threading
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf.config.experimental.set_visible_devices([], 'GPU')

episode_number_max = 1000 * 200
obs_size = 12
done = Manager().Value('i', 0)
run_name = '1'
trainer_count = 1
db_start = 1
train_embedded = True
get_model = False
random_action_percentage = 0.1
generate_random = False
actor_layers = None
critic_layers = None
actor_optimizer = 'sgd'
critic_optimizer = 'sgd'
logger = get_logger()
logger.setLevel(level=logging.CRITICAL)


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
        if arg == 'random_action_percentage':
            random_action_percentage = float(val)
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
        # self.rl.read_weight('/home/nader/workspace/robo/Cyrus2DBase/scripts/rl_server/res/i_20230206143523/9/_agent_actor_w.h5', '/home/nader/workspace/robo/Cyrus2DBase/scripts/rl_server/res/i_20230206143523/9/_agent_critic_w.h5')
        self.rd = RedisServer(db_number)
        self.rd.client.flushdb()
        self.episode_com_rewards = []
        self.episode_res = []
        self.latest_episode_rewards = []
        self.raw_data: dict[int, StepData] = dict()
        self.raw_data_lock = Manager().Lock()
        self.cycle = -1
        self.out_path = os.path.join('res', run_name + '_' + start_time, str(db_number))
        os.makedirs(self.out_path, exist_ok=True)
        self.episode_number = Manager().Value('i', 0)
        self.step_number = Manager().Value('i', 0)
        self.local_done = Manager().Value('i', 0)
        self.init_trainer()

    def add_trainer_info(self, pre_num_cycle, values):
        with self.raw_data_lock:
            cycle = int(pre_num_cycle.split('_')[-1])
            is_done = int(values[0]) >= 2
            is_start = int(values[0]) == 0
            status = int(values[0])
            if is_start:
                return is_start, is_done, status, 0
            reward_cycle = cycle - 1
            if reward_cycle not in self.raw_data.keys():
                self.raw_data[reward_cycle] = StepData()
            self.raw_data[reward_cycle].done = is_done
            self.raw_data[reward_cycle].reward = values[1]
            if is_done:
                self.raw_data[reward_cycle].next_state = None
            return is_start, is_done, status, values[1]

    def add_player_info(self, pre_num_cycle, values):
        with self.raw_data_lock:
            cycle = int(pre_num_cycle.split('_')[-1])
            if cycle not in self.raw_data.keys():
                self.raw_data[cycle] = StepData()
            self.raw_data[cycle].state = values
            cycle -= 1
            if cycle not in self.raw_data.keys():
                self.raw_data[cycle] = StepData()
            if self.raw_data[cycle].done is False:
                self.raw_data[cycle].next_state = values

    def add_player_action(self, pre_num_cycle, action):
        with self.raw_data_lock:
            cycle = int(pre_num_cycle.split('_')[-1])
            if cycle not in self.raw_data.keys():
                self.raw_data[cycle] = StepData()
            self.raw_data[cycle].action = action

    def add_data_to_buffer(self, current_cycle):
        with self.raw_data_lock:
            should_remove = []
            for key in self.raw_data.keys():
                if self.raw_data[key].next_state is None and self.raw_data[key].done is False:
                    if key < current_cycle - 20:
                        should_remove.append(key)
                    continue
                if self.raw_data[key].reward is None:
                    if key < current_cycle - 20:
                        should_remove.append(key)
                    continue
                if self.raw_data[key].state is None:
                    if key < current_cycle - 20:
                        should_remove.append(key)
                    continue
                if self.raw_data[key].action is None:
                    if key < current_cycle - 20:
                        should_remove.append(key)
                    continue
                self.shared_buffer.add(Transition(self.raw_data[key].state, self.raw_data[key].action, self.raw_data[key].reward, self.raw_data[key].next_state))
                # self.rl.add_to_buffer(Transition(self.data[key].state, self.data[key].action, self.data[key].reward, self.data[key].next_state))
                should_remove.append(key)
            for key in should_remove:
                del self.raw_data[key]

    def end_function(self):
        logger.critical('saving data')
        f = open(os.path.join(self.out_path, 'episode_com_rewards'), 'w')
        f.write('\n'.join([str(i) for i in self.episode_com_rewards]))
        f = open(os.path.join(self.out_path, 'episode_res'), 'w')
        f.write('\n'.join([str(i) for i in self.episode_res]))

    def player_end_function(self):
        self.rl.save_weight(self.out_path)
        # self.shared_buffer.save_to_file(self.out_path)

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
        pre_num_cycle, values = self.rd.get_msg_from_no_cycle(num=0, msg_length=[2], wait_time_second=1, done=done)
        logger.info(f'trainer received({self.cycle}): {pre_num_cycle}, {values}')
        if pre_num_cycle is None:
            logger.critical('Did not receive any message from trainer!')
            return False
        else:
            self.cycle = int(pre_num_cycle.split('_')[-1])
            self.step_number.value += 1
            is_start, is_done, status, reward = self.add_trainer_info(pre_num_cycle, values)
            logger.info(f'cycle:{self.cycle} is_start:{is_start} is_done:{is_done} status:{status} reward:{reward}')
            if not is_start:
                self.latest_episode_rewards.append(reward)
            logger.info(f'trainer sent ' + f'{pre_num_cycle}' + ', OK')
            self.rd.set_msg(pre_num_cycle, 'OK')
            if is_start:  # start
                self.rl.random_process.reset_states()
            elif is_done:  # end
                self.episode_res.append(status)
                self.episode_com_rewards.append(sum(self.latest_episode_rewards))
                self.latest_episode_rewards = []
                self.episode_number.value += 1
        return True

    def run_player_one_step(self):
        pre_num_cycle, msg = self.rd.get_msg_from_no_cycle(num=1, msg_length=[obs_size], wait_time_second=0.5, done=done)
        logger.warning(f'player received({self.cycle}): {pre_num_cycle}, {msg}')
        if pre_num_cycle is not None:
            if isinstance(msg, str):  # FAKE message
                logger.warning(f'player sent ' + f'{pre_num_cycle}' + ', OK' + ' (Fake MSG)')
                self.rd.set_msg(pre_num_cycle, "OK")
                return False
            else:
                logger.warning(decode_obs(msg))
                self.add_player_info(pre_num_cycle, msg)
                action_arr = self.rl.get_random_action(msg, random_action_percentage, generate_random)
                # logger.warning(f'q: {self.rl.get_q(msg, action_arr)}')
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
                self.add_player_action(pre_num_cycle, action)
                return True
        return False

    def run_trainer(self):
        while True:
            if self.episode_number.value % 1000 == 0:
                self.end_function()
            if self.episode_number.value == episode_number_max or done.value == 1:
                self.end_function()
                break
            if not self.run_trainer_one_step():
                break
        self.local_done.value = 1

    def run_player(self):
        while True:
            if self.local_done.value == 1:
                self.player_end_function()
                break
            if self.step_number.value % 50000 == 0:
                self.player_end_function()
            res = self.run_player_one_step()
            if res:
                self.add_data_to_buffer(self.cycle)
                if self.train_embedded:
                    self.rl.update(32)

    def run(self):
        logger.critical('Start')

        trainer_process = threading.Thread(target=self.run_trainer)
        trainer_process.start()

        self.run_player()

        trainer_process.join()

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

