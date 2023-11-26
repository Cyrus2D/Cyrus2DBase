
# BEGIN: Generate grpc python code
python3 -m grpc_tools.protoc -I../../../protos --python_out=. --pyi_out=. --grpc_python_out=. ../../../protos/cyrus.proto
# END: Generate grpc python code
