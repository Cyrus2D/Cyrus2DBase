import time
import redis


class RedisServer:
    FROM_AGENT_PRE_POSE = 'req'
    FROM_SERVER_PRE_POSE = 'resp'

    def __init__(self, db_number):
        self.client = redis.Redis(db=db_number)
        self.read_keys = []

    def get_msg_from(self, num, msg_length=None, cycle=None, wait_time_second=None, done=None):
        start_with = RedisServer.FROM_AGENT_PRE_POSE + '_' + str(num) + (('_' + str(cycle)) if cycle is not None else '')
        start_time = time.time()
        while True:
            if done is not None and done.value == 1:
                break
            if wait_time_second:
                waited_time = time.time() - start_time
                if waited_time > wait_time_second:
                    break
            if self.client.exists(start_with):
                msg_type = str(self.client.type(start_with).decode())
                if msg_type == 'string':
                    msg = str(self.client.get(start_with).decode())
                    self.client.delete(start_with)
                    try:
                        msg = int(msg)
                    except:
                        pass
                    return start_with, msg
                elif msg_type == 'list':
                    msg = self.client.lrange(start_with, 0, -1)
                    if len(msg) in msg_length:
                        msg_list = [float(v.decode()) for v in msg]
                        self.client.delete(start_with)
                        return start_with, msg_list
            time.sleep(0.0001)
            # keys: list = self.client.keys()
            # if cycle is not None:
            #     msg = self.client.lrange(start_with, 0, -1)
            #     if len(msg) != 0:
            #         if len(msg) in msg_length:
            #             msg_list = [float(v.decode()) for v in msg]
            #             self.client.delete(start_with)
            #             return start_with, msg_list
            # else:
            #     for k in keys:
            #         key = k.decode()
            #         if not key.startswith(start_with):
            #             continue
            #         msg = self.client.lrange(key, 0, -1)
            #         if len(msg) != 0:
            #             if len(msg) in msg_length:
            #                 msg_list = [float(v.decode()) for v in msg]
            #                 self.client.delete(key)
            #                 return key, msg_list
            #         break
            # time.sleep(0.0001)
        return None, None

    def set_msg(self, key, message):
        response_key = key.replace(RedisServer.FROM_AGENT_PRE_POSE, RedisServer.FROM_SERVER_PRE_POSE)
        res = 0
        while res <= 0:
            if isinstance(message, int) or isinstance(message, str):
                res = self.client.set(response_key, message)
            elif isinstance(message, list):
                res = self.client.rpush(response_key, *message)
            if res <= 0:
                self.client.delete(response_key)