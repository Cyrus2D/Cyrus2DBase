# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import cyrus_pb2 as cyrus__pb2


class SampleServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetBestAction = channel.unary_unary(
                '/cyrus.SampleService/GetBestAction',
                request_serializer=cyrus__pb2.StateMessage.SerializeToString,
                response_deserializer=cyrus__pb2.Action.FromString,
                )
        self.SetTrainerRequest = channel.unary_unary(
                '/cyrus.SampleService/SetTrainerRequest',
                request_serializer=cyrus__pb2.TrainerRequest.SerializeToString,
                response_deserializer=cyrus__pb2.OK.FromString,
                )


class SampleServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetBestAction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetTrainerRequest(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SampleServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetBestAction': grpc.unary_unary_rpc_method_handler(
                    servicer.GetBestAction,
                    request_deserializer=cyrus__pb2.StateMessage.FromString,
                    response_serializer=cyrus__pb2.Action.SerializeToString,
            ),
            'SetTrainerRequest': grpc.unary_unary_rpc_method_handler(
                    servicer.SetTrainerRequest,
                    request_deserializer=cyrus__pb2.TrainerRequest.FromString,
                    response_serializer=cyrus__pb2.OK.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'cyrus.SampleService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SampleService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetBestAction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cyrus.SampleService/GetBestAction',
            cyrus__pb2.StateMessage.SerializeToString,
            cyrus__pb2.Action.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetTrainerRequest(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cyrus.SampleService/SetTrainerRequest',
            cyrus__pb2.TrainerRequest.SerializeToString,
            cyrus__pb2.OK.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
