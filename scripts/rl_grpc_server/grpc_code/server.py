import cyrus_pb2_grpc as pb2_grpc
import cyrus_pb2 as pb2

import google.protobuf
from concurrent import futures
import grpc

class SampleService(pb2_grpc.SampleServiceServicer):
    def __init__(self):
        pass

    def GetBestAction(self, request, context):
        # Implement your logic here to get the best action based on the given state
        print(f'{request}')
        # Your code here to calculate the best action
        best_action = pb2.Action(Dash=pb2.ActionDash(Power=1, Dir=pb2.Ang2D(Angle=1)), Turn=pb2.ActionTurn(Dir=pb2.Ang2D(Angle=1)))
        return best_action

    def SetReward(self, request, context):
        # Implement your logic here to set the reward based on the given reward value
        print(f'{request}')
        # Your code here to set the reward
        return pb2.OK()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SampleServiceServicer_to_server(SampleService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()