import socket 
from threading import Thread 
from socketserver import ThreadingMixIn 
import utils
import json
import random
import multiprocessing as mp

'''
Marshalling protocol
0: base generation or noise matrix generation boolean
1: current seed
2: current iteration
3: quit boolean
'''

fitnesses = []
best_weights = None

def on_new_client(conn,addr):
    i = 0 
    num_iters = 3 # int(input("Enter the number of iterations."))
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
    
    print("listen")
    tcpServer.listen(4) 
    print("Multithreaded Python server: Waiting for connections from TCP clients...") 
    print("accept")
    conn, addr = tcpServer.accept()
    Thread(target=on_new_client,args=(conn,addr)).start()
    print("threaded")