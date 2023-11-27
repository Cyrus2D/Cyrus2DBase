import grpc
import cyrus_pb2_grpc as pb2_grpc
import cyrus_pb2 as pb2


class CyrusClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.SampleServiceStub(self.channel)

    def GetBestAction(self, state):
        """
        Client function to call the rpc for GetBestAction
        """
        state = pb2.State(Position=pb2.Vec2D(X=1, Y=2), Body=pb2.Ang2D(Angle=3), Cycle=4)
        action = self.stub.GetBestAction(state)
        return action
    
    def SetReward(self, reward):
        """
        Client function to call the rpc for SetReward
        """
        reward = pb2.Reward(Value=1, Cycle=2, Unum=3)
        response = self.stub.SetReward(reward)
        return response


if __name__ == '__main__':
    client = CyrusClient()

    state = pb2.State(Position=pb2.Vec2D(X=1, Y=2), Body=pb2.Ang2D(Angle=3), Cycle=4)
    action = client.GetBestAction(state)
    print(f'{action}')
    reward = client.SetReward(pb2.Reward(Value=1, Cycle=2, Unum=3))
    print(f'{reward}')