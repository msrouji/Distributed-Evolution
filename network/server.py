import socket 
from threading import Thread 
from socketserver import ThreadingMixIn 
import utils
import json
import random

'''
Marshalling protocol

0: base generation or noise matrix generation boolean
1: current seed
2: current iteration
3: quit boolean
'''

fitnesses = []
best_weights = None

# Multithreaded Python server : TCP Server Socket Thread Pool
class ClientThread(Thread): 
 
    def __init__(self,ip,port): 
        Thread.__init__(self) 
        self.ip = ip 
        self.port = port 
        print("[+] New server socket thread started for " + ip + ":" + str(port))
 
    def run(self):
        i = 0 
        num_iters = 10 # int(input("Enter the number of iterations."))
        json_msg = {"quit": False}
        json_msg["gen"] = True
        json_msg["seed"] = random.randint(0,100000)
        json_msg["iter"] = 0
        json_msg["weights"] = None
        message = json.dumps(json_msg)
        conn.send(utils.stringToBytes(message))
        data = ""
        while True: 
            raw = conn.recv(2048)
            data += utils.stringFromBytes(raw)
            if data!=None and data!="":
                if data[-1]=="}":
                    if True: # later: if have x% of clients
                        i+=1
                    client_data = json.loads(data)
                    data = ""
                    display = client_data.copy()
                    display["weights"] = "weights"
                    print("Server received data:", display)
                    json_msg = {"quit": False}
                    json_msg["gen"] = True
                    json_msg["seed"] = random.randint(0,100000)
                    json_msg["iter"] = i
                    fitnesses.append(client_data["fitness"])
                    best_weights = client_data["weights"]
                    json_msg["weights"] = best_weights
                    if i > num_iters:
                        json_msg["quit"] = True
                    message = json.dumps(json_msg)
                    
                    conn.send(utils.stringToBytes(message))
                    if i > num_iters:
                        break


# Multithreaded Python server: TCP Server Socket Program Stub
TCP_IP = "127.0.0.1"
TCP_PORT = 3000
BUFFER_SIZE = 1024

tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
tcpServer.bind((TCP_IP, TCP_PORT)) 
threads = [] 
 
while True: 
    tcpServer.listen(4) 
    print("Multithreaded Python server: Waiting for connections from TCP clients...") 
    (conn, (ip,port)) = tcpServer.accept()
    ct = ClientThread(ip,port)
    Thread(target=ct.run(), args=((conn, (ip,port)),)).start()

# for t in threads: 
#     t.join() 

### Server from tutorial

# import sys
# import socket
# import selectors
# import types
# import utils

# sel = selectors.DefaultSelector()

# def accept_wrapper(sock):
#     conn, addr = sock.accept()  # Should be ready to read
#     print("accepted connection from", addr)
#     conn.setblocking(False)
#     data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
#     events = selectors.EVENT_READ | selectors.EVENT_WRITE
#     sel.register(conn, events, data=data)

# def service_connection(key, mask):
#     sock = key.fileobj
#     data = key.data
#     if mask & selectors.EVENT_READ:
#         recv_data = sock.recv(1024)  # Should be ready to read
#         if recv_data:
#             data.outb += recv_data
#         else:
#             print("closing connection to", data.addr)
#             sel.unregister(sock)
#             sock.close()
#     if mask & selectors.EVENT_WRITE:
#         if data.outb:
#             print(utils.floatFromBytes(data.outb)) #utils.floatFromBytes()
#             print("echoing", repr(data.outb), "to", data.addr)
#             sent = sock.send(data.outb)  # Should be ready to write
#             data.outb = data.outb[sent:]

# if len(sys.argv) != 3:
#     print("usage:", sys.argv[0], "<host> <port>")
#     sys.exit(1)

# host, port = sys.argv[1], int(sys.argv[2])
# lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# lsock.bind((host, port))
# lsock.listen()
# print("listening on", (host, port))
# lsock.setblocking(False)
# sel.register(lsock, selectors.EVENT_READ, data=None)

# try:
#     while True:
#         events = sel.select(timeout=None)
#         for key, mask in events:
#             if key.data is None:
#                 accept_wrapper(key.fileobj)
#             else:
#                 service_connection(key, mask)
# except KeyboardInterrupt:
#     print("caught keyboard interrupt, exiting")
# finally:
#     sel.close()


### Simple Server

# import selectors
# import socket
# import utils

# sel = selectors.DefaultSelector()

# def accept(sock, mask):
#     conn, addr = sock.accept()
#     print('accepted', conn, 'from', addr)
#     conn.setblocking(False)
#     sel.register(conn, selectors.EVENT_READ, read)

# def read(conn, mask):
#     data = conn.recv(1000)
#     if data:
#         print(type(data))
#         print('echoing', utils.floatFromBytes(eval(repr(data))), 'to', conn)
#         conn.send(data)
#     else:
#         print('closing', conn)
#         sel.unregister(conn)
#         conn.close()

# sock = socket.socket()
# sock.bind(('localhost', 3000))
# sock.listen(100)
# sock.setblocking(False)
# sel.register(sock, selectors.EVENT_READ, accept)

# while True:
#     events = sel.select()
#     for key, mask in events:
#         callback = key.data
#         callback(key.fileobj, mask)