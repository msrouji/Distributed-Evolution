import socket
import utils

host = socket.gethostname() 
port = 3000
BUFFER_SIZE = 2000 
MESSAGE = input("Client: Enter message/ Enter exit:")
MESSAGE = utils.stringToBytes(MESSAGE)
 
tcpClientA = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
tcpClientA.connect((host, port))

while MESSAGE != 'exit':
    tcpClientA.send(MESSAGE)     
    data = tcpClientA.recv(BUFFER_SIZE)
    print(" Client received data:", data)
    MESSAGE = input("Client: Enter message to continue/ Enter exit:")
    MESSAGE = utils.stringToBytes(MESSAGE)

tcpClientA.close() 


### Client from tutorial

# #!/usr/bin/env python3

# import sys
# import socket
# import selectors
# import types
# import utils

# sel = selectors.DefaultSelector()

# def start_connections(host, port, num_conns, messages):
#     server_addr = (host, port)
#     for i in range(0, num_conns):
#         connid = i + 1
#         print("starting connection", connid, "to", server_addr)
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.setblocking(False)
#         sock.connect_ex(server_addr)
#         events = selectors.EVENT_READ | selectors.EVENT_WRITE
#         data = types.SimpleNamespace(
#             connid=connid,
#             msg_total=sum(len(m) for m in messages),
#             recv_total=0,
#             messages=list(messages),
#             outb=b"",
#         )
#         sel.register(sock, events, data=data)

# def service_connection(key, mask, messages):
#     sock = key.fileobj
#     data = key.data
#     if mask & selectors.EVENT_READ:
#         recv_data = sock.recv(1024)  # Should be ready to read
#         if recv_data:
#             print("received", utils.floatFromBytes(eval(repr(recv_data))), "from connection", data.connid)
#             data.recv_total += len(recv_data)
#         if not recv_data or data.recv_total == data.msg_total:
#             print("closing connection", data.connid)
#             sel.unregister(sock)
#             sock.close()
#     if mask & selectors.EVENT_WRITE:
#         if not data.outb and data.messages:
#             data.outb = data.messages.pop(0)
#         if data.outb:
#             print("sending", repr(data.outb), "to connection", data.connid)
#             sent = sock.send(data.outb)  # Should be ready to write
#             data.outb = data.outb[sent:]
            


# # if len(sys.argv) != 4:
# #     print("usage:", sys.argv[0], "<host> <port> <num_connections>")
# #     sys.exit(1)

# def setupClient(messages, host, port, num_conns):
#     start_connections(host, int(port), int(num_conns), messages)
#     try:
#         while True:
#             events = sel.select(timeout=1)
#             if events:
#                 for key, mask in events:
#                     service_connection(key, mask, messages)
#             if not sel.get_map(): # Check for a socket being monitored to continue.
#                 break
#     except KeyboardInterrupt:
#         print("caught keyboard interrupt, exiting")
#     finally:
#         sel.close()


# #### Uncomment for debugging
# # msg=0.4536
# # messages=[utils.floatToBytes(msg)]

# # start_connections(host="127.0.0.1", port=3000, num_conns=1, messages=[utils.floatToBytes(msg)])
# # try:
# #     while True:
# #         events = sel.select(timeout=1)
# #         if events:
# #             for key, mask in events:
# #                 service_connection(key, mask, messages)
# #         if not sel.get_map(): # Check for a socket being monitored to continue.
# #             break
# # except KeyboardInterrupt:
# #     print("caught keyboard interrupt, exiting")
# # finally:
# #     sel.close()