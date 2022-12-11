import time

from redis_server import RedisServer


class StepData:
    def __init__(self):
        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None
        self.done = False

from ddpg import DeepAC
rl = DeepAC()
rl.create_model_actor_critic()

rd = RedisServer()
rd.client.flushdb()

data: dict[int, StepData] = {}


def add_trainer_info(pre_num_cycle, values):
    cycle = int(pre_num_cycle.split('_')[-1])
    is_done = int(values[0]) >= 2
    is_start = int(values[0]) == 0
    status = int(values[0])
    if is_start:
        return is_start, is_done, status, 0
    reward_cycle = cycle - 1
    if reward_cycle not in data.keys():
        data[reward_cycle] = StepData()
    data[reward_cycle].done = is_done
    data[reward_cycle].reward = values[1]
    if is_done:
        data[reward_cycle].next_state = None
    return is_start, is_done, status, data[reward_cycle].reward


def add_player_info(pre_num_cycle, values):
    cycle = int(pre_num_cycle.split('_')[-1])
    if cycle not in data.keys():
        data[cycle] = StepData()
    data[cycle].state = values
    cycle -= 1
    if cycle not in data.keys():
        data[cycle] = StepData()
    if data[cycle].done is False:
        data[cycle].next_state = values


def add_player_action(pre_num_cycle, action):
    cycle = int(pre_num_cycle.split('_')[-1])
    if cycle not in data.keys():
        data[cycle] = StepData()
    data[cycle].action = action


def add_data_to_buffer(current_cycle):
    global data
    should_remove = []
    for key in data.keys():
        if data[key].next_state is None and data[key].done is False:
            if key < current_cycle - 2:
                should_remove.append(key)
            continue
        if data[key].reward is None:
            if key < current_cycle - 2:
                should_remove.append(key)
            continue
        if data[key].state is None:
            if key < current_cycle - 2:
                should_remove.append(key)
            continue
        if data[key].action is None:
            if key < current_cycle - 2:
                should_remove.append(key)
            continue
        rl.add_to_buffer(data[key].state, data[key].action, data[key].reward, data[key].next_state)
        should_remove.append(key)
    for key in should_remove:
        del data[key]



patch_number = 0
patch_number_max = 1000
train_episode_number = 0
train_episode_number_max = 100
test_episode_number = 0
test_episode_number_max = 10
is_train = True
received_messages = {}
player_ignore_cycles = []
training_cycles = []
testing_cycles = []
training_step_rewards = []
testing_step_rewards = []
training_episode_last_rewards = []
testing_episode_last_rewards = []
training_episode_com_rewards = []
testing_episode_com_rewards = []
training_episode_res = []
testing_episode_res = []
latest_episode_rewards = []

def end_function():
    print('EEEEEEEEEEEENNNNNNNNNDDDDDDDDDDD')
    f = open('player_ignore_cycles', 'w')
    f.write('\n'.join([str(i) for i in player_ignore_cycles]))
    f = open('training_cycles', 'w')
    f.write('\n'.join([str(i) for i in training_cycles]))
    f = open('testing_cycles', 'w')
    f.write('\n'.join([str(i) for i in testing_cycles]))
    f = open('training_step_rewards', 'w')
    f.write('\n'.join([str(i) for i in training_step_rewards]))
    f = open('testing_step_rewards', 'w')
    f.write('\n'.join([str(i) for i in testing_step_rewards]))
    f = open('training_episode_last_rewards', 'w')
    f.write('\n'.join([str(i) for i in training_episode_last_rewards]))
    f = open('testing_episode_last_rewards', 'w')
    f.write('\n'.join([str(i) for i in testing_episode_last_rewards]))
    f = open('training_episode_com_rewards', 'w')
    f.write('\n'.join([str(i) for i in training_episode_com_rewards]))
    f = open('2', 'w')
    f.write('\n'.join([str(i) for i in testing_episode_com_rewards]))
    f = open('training_episode_res', 'w')
    f.write('\n'.join([str(i) for i in training_episode_res]))
    f = open('testing_episode_res', 'w')
    f.write('\n'.join([str(i) for i in testing_episode_res]))
    f = open('latest_episode_rewards', 'w')
    f.write('\n'.join([str(i) for i in latest_episode_rewards]))
    while True:
        time.sleep(1)


i = 0
cycle = None
while True:
    # print('#'*100, cycle)
    pre_num_cycle, values = rd.get_from_wait(0, [2], cycle, 1)
    if pre_num_cycle is None and cycle is None:
        continue
    if pre_num_cycle is None:
        rd.set(RedisServer.FROM_AGENT_PRE_POSE + '_' + str(0) + '_' + str(cycle), [1])
    if pre_num_cycle is not None:
        # print(pre_num_cycle, values)
        cycle = int(pre_num_cycle.split('_')[-1])
        is_start, is_done, status, reward = add_trainer_info(pre_num_cycle, values)
        if not is_start:
            # if is_train:
            #     training_step_rewards.append(reward)
            # else:
            #     testing_step_rewards.append(reward)
            latest_episode_rewards.append(reward)
        received_messages[pre_num_cycle] = values
        rd.set(pre_num_cycle, [1])
        if is_start:  # start
            pass
        elif is_done:  # end
            if is_train:
                training_episode_last_rewards.append(reward)
                training_episode_res.append(status)
                training_episode_com_rewards.append(sum(latest_episode_rewards))
                latest_episode_rewards = []
                train_episode_number += 1
                if train_episode_number == train_episode_number_max:
                    is_train = False
                    train_episode_number = 0
            else:
                testing_episode_last_rewards.append(reward)
                testing_episode_res.append(status)
                testing_episode_com_rewards.append(sum(latest_episode_rewards))
                print(testing_episode_com_rewards[-1])
                latest_episode_rewards = []
                test_episode_number += 1
                if test_episode_number == test_episode_number_max:
                    is_train = True
                    test_episode_number = 0
                    patch_number += 1
            if patch_number == patch_number_max:
                end_function()
    # if is_train:
    #     training_cycles.append(cycle)
    # else:
    #     testing_cycles.append(cycle)

    pre_num_cycle, values = rd.get_from_wait(1, [6, 1], cycle, wait_time_second=0.5)
    if pre_num_cycle is not None:
        if len(values) == 1:  # Fake
            rd.set(pre_num_cycle, [0])
        else:
            # print(pre_num_cycle, values)
            add_player_info(pre_num_cycle, values)
            action_arr = rl.get_random_action(values, patch_number, patch_number_max, None if is_train else 0.0)
            action_tmp = action_arr.tolist()
            action = []
            for a in action_tmp:
                action.append(float(a))
            # print('###################################')
            # print(type(action), action)
            rd.set(pre_num_cycle, action)
            add_player_action(pre_num_cycle, action)
            received_messages[pre_num_cycle] = values
    else:
        player_ignore_cycles.append(cycle)
    i += 1
    add_data_to_buffer(cycle)
    cycle += 1

# d = StepData()
# d.state = [1,2,3]
# d.next_state = [1,2,3]
# d.reward = 1
# d.action = [5]
# for i in range(1000):
#     data[i] = d
# add_data_to_buffer()
#
#
# pass