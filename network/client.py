import socket
import utils
import json
import time
import pickle
import codecs

from evolution.OpenAIGym import *
from evolution.evostra import FeedForwardNetwork
from evolution.evostra import CNN2D
import numpy as np

host = "127.0.0.1" 
port = 3000
BUFFER_SIZE = 2048
while True:
    try:
        tcpClientA = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        tcpClientA.connect((host, port))
        break
    except:
        pass

quit_msg = False

agent = Agent("LunarLander-v2",
                3,
                FeedForwardNetwork,
                [16, 16],
                population_size=40,
                sigma=0.1,
                learning_rate=0.01,
                decay=0.999,
                num_threads=6,
                eps_avg=1,
                wait=1.0,
                avg_runs=1,
                print_step=1,
                save_step=None)

data = ""

while not quit_msg:  
    raw = tcpClientA.recv(BUFFER_SIZE)
    data += utils.stringFromBytes(raw)
    if data!=None and data!="":
        if data[-1]=="}":
            server_data = json.loads(data)
            display = server_data.copy()
            display["weights"] = "weights"
            print("Client received data:", display)
            data = ""
            weights = server_data["weights"]
            if weights is not None:
                agent.set_weights(pickle.loads(codecs.decode(weights.encode(), "base64")))
            agent.train(20)
            fitness = agent.play_episodes(2,render=False)
            weights = agent.get_weights()
            pickled_weights = codecs.encode(pickle.dumps(weights), "base64").decode()
            json_msg = {"seed":server_data["seed"], "fitness":fitness, "iter":server_data["iter"], "weights":pickled_weights}
            
            quit_msg = server_data["quit"]
            message = json.dumps(json_msg)
            message = utils.stringToBytes(message)
            tcpClientA.send(message)   
    