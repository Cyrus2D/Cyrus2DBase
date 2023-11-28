import time
from table import QTable
import threading
import cyrus_pb2_grpc as pb2_grpc
import cyrus_pb2 as pb2
from ddpg import DeepAC
from threading import RLock

lock = RLock()


class State:
    def __init__(self, grpcState: pb2.StateMessage):
        self.grpcState = grpcState
        self.rawState = None
        self.ConvertToRawState()
    
    def ConvertToRawState(self):
        if self.grpcState.WhichOneof('State') == 'GoToBallState':
            self.rawState = [int((self.grpcState.GoToBallState.BodyDiff + 180.0) / 10.0)]
        elif self.grpcState.WhichOneof('State') == 'RawList':
            # self.rawState = [int((self.grpcState.RawList.Value[0] + 180.0) / 10.0)]
            self.rawState = [(self.grpcState.RawList.Value[0] + 180.0) / 360.0]
        elif self.grpcState.WhichOneof('State') == 'GoToPointState':
            self.rawState = [int((self.grpcState.GoToPointState.BodyDiff + 180.0) / 10.0)]
            

class Action:
    def __init__(self, rawAction, grpcActionType):
        self.rawAction = rawAction
        self.grpcAction = None
        self.grpcActionType = grpcActionType
        self.ConvertToGrpcAction()
    
    def ConvertToGrpcAction(self):
        if self.grpcActionType == 'Dash':
            self.grpcAction = pb2.Action(Dash=pb2.ActionDashMessage(Power=100, Dir=self.rawAction * 180))
        elif self.grpcActionType == 'Turn':
            self.grpcAction = pb2.Action(Turn=pb2.ActionTurnMessage(Dir=self.rawAction * 10))

class Results:
    def __init__(self):
        self.allRewards = []
        self.allEpisodesRewards = []
        self.allEndReards = []
        self.allStatus = []
        self.lastEpisodeRewards = []

    def SaveResult(self):
        with open('allRewards.txt', 'a') as f:
            strAllRewards = ''
            for reward in self.allRewards:
                strAllRewards += str(reward) + '\n'
            f.write(strAllRewards)
        with open('allEpisodesRewards.txt', 'a') as f:
            strAllEpisodesRewards = ''
            for reward in self.allEpisodesRewards:
                strAllEpisodesRewards += str(reward) + '\n'
            f.write(strAllEpisodesRewards)
        with open('allEndReards.txt', 'a') as f:
            strAllEndReards = ''
            for reward in self.allEndReards:
                strAllEndReards += str(reward) + '\n'
            f.write(strAllEndReards)
        with open('allStatus.txt', 'a') as f:
            strAllStatus = ''
            for status in self.allStatus:
                strAllStatus += str(status) + '\n'
            f.write(strAllStatus)

    def AddResult(self, reward, done):
        self.allRewards.append(reward)
        self.lastEpisodeRewards.append(reward)
        if done:
            self.allEpisodesRewards.append(sum(self.lastEpisodeRewards))
            self.lastEpisodeRewards = []
            self.allEndReards.append(reward)
            self.allStatus.append(done)
        if len(self.allEndReards) % 100 == 0:
            self.SaveResult()
            self.allRewards = []
            self.allEpisodesRewards = []
            self.allEndReards = []
            self.allStatus = []
            
            
class StepPreData:
    def __init__(self):
        self.state: list[float] = None
        self.next_state: list[float] = None
        self.reward: float = None
        self.action: list[float] = None
        self.done: bool = False #TODO
        self.data_number = 0
    
    def AddState(self, state):
        self.state = state
        self.data_number += 1
        return self.CheckComplete()
    
    def AddNextState(self, next_state):
        self.next_state = next_state
        self.data_number += 1
        return self.CheckComplete()
    
    def AddReward(self, reward):
        self.reward = reward
        self.data_number += 1
        return self.CheckComplete()

    def AddAction(self, action):
        self.action = action
        self.data_number += 1
        return self.CheckComplete()

    def CheckComplete(self):
        return self.data_number == 4

class Table:
    def __init__(self):
        # self.rl = QTable()
        self.observation_size = 1
        self.action_size = 1
        self.rl = DeepAC(observation_size=self.observation_size, action_size=self.action_size)
        self.rl.create_model_actor_critic()
        self.data: dict[int, StepPreData] = {}
        self.results = Results()


    def AddTrainerInfo(self, trainerRequest: pb2.TrainerRequest):
        with lock:
            if trainerRequest.Start:
                # print('Add trainer info Start', trainerRequest.Cycle, trainerRequest.Start)
                return # remove data from data
            # print('Add trainer info', trainerRequest.Cycle, trainerRequest.Reward)
            cycle = trainerRequest.Cycle - 1
            if cycle not in self.data.keys():
                self.data[cycle] = StepPreData()
            self.data[cycle].done = trainerRequest.Done
            complete = self.data[cycle].AddReward(trainerRequest.Reward)
            if complete:
                self.AddDataToBuffer(cycle)
            self.results.AddResult(trainerRequest.Reward, trainerRequest.Done)

    def AddPlayerInfo(self, playerState: pb2.StateMessage):
        with lock:
            cycle = playerState.Cycle
            if cycle not in self.data.keys():
                    self.data[cycle] = StepPreData()
            state = State(playerState)
            complete = self.data[cycle].AddState(state.rawState)
            if complete:
                self.AddDataToBuffer(cycle)
            cycle = playerState.Cycle - 1
            if cycle not in self.data.keys():
                self.data[cycle] = StepPreData()
            complete = self.data[cycle].AddNextState(state.rawState)
            if complete:
                self.AddDataToBuffer(cycle)
            
    def AddPlayerAction(self, cycle, action):
        # print('add player action', cycle, action)
        if cycle not in self.data.keys():
            self.data[cycle] = StepPreData()
        complete = self.data[cycle].AddAction(action)
        if complete:
            self.AddDataToBuffer(cycle)

    def AddDataToBuffer(self, cycle):
        # print('>>add data to buffer', cycle, self.data[cycle].state, self.data[cycle].action, self.data[cycle].reward, self.data[cycle].next_state)
        self.rl.add_to_buffer(self.data[cycle].state, self.data[cycle].action, self.data[cycle].reward, self.data[cycle].next_state)
        del self.data[cycle]

    def GetAction(self, grpcState: pb2.StateMessage):
        with lock:
            epsilon = 0.1
            state = State(grpcState).rawState
            rawAction = self.rl.GetRandomBestAction(state, epsilon)
            self.AddPlayerAction(grpcState.Cycle, rawAction)
            action = Action(rawAction, 'Dash')
            return action.grpcAction
