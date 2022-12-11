import time
import redis


class RedisServer:
    FROM_AGENT_PRE_POSE = 'req'
    FROM_SERVER_PRE_POSE = 'resp'

    def __init__(self):
        self.client = redis.Redis()
        self.read_keys = []

    def get_from_wait(self, num, msg_length, cycle=None, wait_time_second=None):
        start_with = RedisServer.FROM_AGENT_PRE_POSE + '_' + str(num) + (('_' + str(cycle)) if cycle is not None else '')
        # print(start_with)
        start_time = time.time()
        while True:
            keys: list = self.client.keys()
            if wait_time_second:
                waited_time = time.time() - start_time
                if waited_time > wait_time_second:
                    break
            if cycle is not None:
                msg = self.client.lrange(start_with, 0, -1)
                if len(msg) != 0:
                    if len(msg) in msg_length:
                        msg_list = [float(v.decode()) for v in msg]
                        self.client.delete(start_with)
                        return start_with, msg_list
            else:
                for k in keys:
                    key = k.decode()
                    if not key.startswith(start_with):
                        continue
                    msg = self.client.lrange(key, 0, -1)
                    if len(msg) != 0:
                        if len(msg) in msg_length:
                            msg_list = [float(v.decode()) for v in msg]
                            self.client.delete(key)
                            return key, msg_list
                    break
            time.sleep(0.0001)
        return None, None

    def get_from_waita(self, num, msg_length, cycle=None, wait_time_second=None):
        start_with = RedisServer.FROM_AGENT_PRE_POSE + '_' + str(num) + (('_' + str(cycle)) if cycle is not None else '')
        # print(start_with)
        start_time = time.time()
        while True:
            keys: list = self.client.keys()
            if wait_time_second:
                waited_time = time.time() - start_time
                if waited_time > wait_time_second:
                    break
            for k in keys:
                key = k.decode()
                if not key.startswith(start_with):
                    continue
                request = self.client.lrange(key, 0, -1)
                req_list = [float(v.decode()) for v in request]
                self.client.delete(key)
                return key, req_list
            time.sleep(0.0001)
        return None, None

    def set(self, key, s):
        response_key = key.replace(RedisServer.FROM_AGENT_PRE_POSE, RedisServer.FROM_SERVER_PRE_POSE)
        # print('try set', response_key, s)
        res = 0
        while res <= 0:
            res = self.client.rpush(response_key, *s)
            if res <= 0:
                self.client.delete(response_key)