import cyrus_pb2_grpc as pb2_grpc
import cyrus_pb2 as pb2

import google.protobuf
from concurrent import futures
import grpc
from main_table import Table

from threading import RLock

lock = RLock()


class SampleService(pb2_grpc.SampleServiceServicer):
    def __init__(self):
        self.tabel = Table()

    def GetBestAction(self, request:pb2.StateMessage, context):
        with lock:
            self.tabel.AddPlayerInfo(request)
            action = self.tabel.GetAction(request)
            return action

    def SetTrainerRequest(self, request: pb2.TrainerRequest, context):
        with lock:
            self.tabel.AddTrainerInfo(request)
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