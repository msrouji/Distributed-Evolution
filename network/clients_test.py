import asyncio
import client
import utils
import socket

messages_list = [0.23, 0.52, 935.36, 0.42, 0.88]

async def main():
    for i,msg in enumerate(messages_list):
        byte_msg = [utils.floatToBytes(msg)]
        host, port, num_conns = ("127.0.0.1", 3000, 1)
        client.setupClient(byte_msg, host, port, num_conns)


asyncio.run(main())

# class DistributedClient():
#     def __init__(self, host, port, retryAttempts=10):
#         self.host = host
#         self.port = port
#         self.retryAttempts = retryAttempts
#         self.socket = None

#     def connect(self, attempt=0):
#         self.socket = socket.socket()
#         if attempts < self.retryAttempts:
#             client.setupClient(byte_msg, host, port, num_conns)
#         if connectionFailed:
#             self.connect(attempt+1)

#     def disconnectSocket(self):
#         self.socket = None

#     def sendDataToDB(self, data):
#         print(data)

#     def readData(self):
#         while True:
#             if self.socket is None:
#                 self.connect()