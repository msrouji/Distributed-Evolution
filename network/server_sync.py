import socket 
from threading import Thread 
from socketserver import ThreadingMixIn 
import utils
import json
import random
import multiprocessing as mp
import pickle
import codecs
import numpy as np

'''
Marshalling protocol

0: base generation or noise matrix generation boolean
1: current seed
2: current iteration
3: quit boolean
'''

best_iters = {}
list_weights = {}
fitnesses = []
best_fitnesses = []
best_fitness = -10000000
best_iter_fitness = -10000000
best_weights = None
num_clients = 0
num_iters = 100 # int(input("Enter the number of iterations."))
curr_iter_seen = {i:0 for i in range(num_iters)}
addfitness = True

json_msg = {"quit": False}
json_msg["gen"] = True
json_msg["seed"] = random.randint(0,100000)
json_msg["iter"] = 0
json_msg["weights"] = None

def on_new_client_max(conn,addr):
    i = 0 
    global json_msg
    global best_fitness
    global best_weights
    global num_clients
    global best_fitnesses
    global addfitness
    global best_iter_fitness
    global best_iters
    message = json.dumps(json_msg)
    conn.send(utils.stringToBytes(message))
    data = ""
    
    percent_threshold = 0.99
    while True: 
        try:
            raw = conn.recv(2048)
            data += utils.stringFromBytes(raw)
            if data!=None and data!="":
                if data[-1]=="}":
                    if True: # later: if have x% of clients
                        i+=1
                    curr_iter_seen[i] = curr_iter_seen[i]+1
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
                    print("checking for update")

                    # if client_data["fitness"]>best_iter_fitness:
                    #     best_iter_fitness = client_data["fitness"]
                    # if i not in best_iters:
                    #     best_iters[i] = client_data["fitness"]
                    # elif client_data["fitness"]>best_iters[i]:
                    #     best_iters[i] = client_data["fitness"]
                    # if client_data["fitness"]>best_fitness:
                    #     best_fitness = client_data["fitness"]
                    #     best_weights = client_data["weights"]
                    #     print("new best!")

                    if client_data["fitness"]>best_iter_fitness:
                        best_iter_fitness = client_data["fitness"]
                    if i not in best_iters:
                        best_iters[i] = client_data["fitness"]
                    elif client_data["fitness"]>best_iters[i]:
                        best_iters[i] = client_data["fitness"]

                    if client_data["fitness"]>best_iters[i]:
                        best_fitness = client_data["fitness"]
                        best_weights = client_data["weights"]

                    if i > num_iters:
                        json_msg["quit"] = True
                    print("Client done with iteration")
                    addfitness = True
                    while True:
                        if curr_iter_seen[i]/num_clients > percent_threshold:
                            break
                    if addfitness:
                        best_fitnesses.append(best_iter_fitness)
                        print(best_iters)
                        addfitness = False
                    best_iter_fitness = -10000000
                    json_msg["weights"] = best_weights
                    message = json.dumps(json_msg)
                    conn.send(utils.stringToBytes(message))
                    if i > num_iters:
                        break
        except:
            pass

def on_new_client_lincomb(conn,addr):
    i = 0 
    global json_msg
    global best_fitness
    global best_weights
    global num_clients
    global best_fitnesses
    global addfitness
    global best_iter_fitness
    global best_iters
    global list_weights
    message = json.dumps(json_msg)
    conn.send(utils.stringToBytes(message))
    data = ""
    
    percent_threshold = 0.99
    while True: 
        raw = conn.recv(2048)
        data += utils.stringFromBytes(raw)
        if data!=None and data!="":
            if data[-1]=="}":
                if True: # later: if have x% of clients
                    i+=1
                curr_iter_seen[i] = curr_iter_seen[i]+1
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
                print("checking for update")

                # if client_data["fitness"]>best_iter_fitness:
                #     best_iter_fitness = client_data["fitness"]
                # if i not in best_iters:
                #     best_iters[i] = client_data["fitness"]
                # elif client_data["fitness"]>best_iters[i]:
                #     best_iters[i] = client_data["fitness"]
                # if client_data["fitness"]>best_fitness:
                #     best_fitness = client_data["fitness"]
                #     best_weights = client_data["weights"]
                #     print("new best!")

                if client_data["fitness"]>best_iter_fitness:
                    best_iter_fitness = client_data["fitness"]
                if i not in best_iters:
                    best_iters[i] = client_data["fitness"]
                elif client_data["fitness"]>best_iters[i]:
                    best_iters[i] = client_data["fitness"]
                if i not in list_weights:
                    list_weights[i] = []
                list_weights[i].append({"weights":client_data["weights"],"fitness":client_data["fitness"]})
                

                if client_data["fitness"]>best_iters[i]:
                    best_fitness = client_data["fitness"]
                    best_weights = client_data["weights"]

                if i > num_iters:
                    json_msg["quit"] = True
                print("Client done with iteration")
                addfitness = True
                while True:
                    if curr_iter_seen[i]/num_clients > percent_threshold:
                        break
                if addfitness:
                    best_fitnesses.append(best_iter_fitness)
                    print(best_iters)
                    addfitness = False
                best_iter_fitness = -10000000
                curr_weight = 0

                fitness_scores = [w["fitness"] for w in list_weights[i]]

                fitness_scores = [float(i)/sum(fitness_scores) for i in fitness_scores]

                for j, weight in enumerate(list_weights[i]):
                    ind = j+1
                    wtp = weight["weights"]
                    ft = weight["fitness"]
                    wt = pickle.loads(codecs.decode(wtp.encode(), "base64"))
                    np_wt = np.array(wt)
                    curr_weight = ((ind-1)/float(ind))*curr_weight+(1/float(ind))*np_wt*fitness_scores[j]
                curr_weight = curr_weight.tolist()
                best_weights = codecs.encode(pickle.dumps(curr_weight), "base64").decode()
                
                json_msg["weights"] = best_weights
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
    Thread(target=on_new_client_max,args=(conn,addr)).start()
    print("threaded")
    num_clients+=1
    print("number of clients on network: ",num_clients)