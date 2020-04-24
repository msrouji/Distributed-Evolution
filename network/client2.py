import socket
import utils

host = "127.0.0.1"  
port = 3000
BUFFER_SIZE = 2000 
MESSAGE = input("Client 2: Enter message/ Enter exit:")
MESSAGE = utils.stringToBytes(MESSAGE)
 
tcpClientA = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
tcpClientA.connect((host, port))

while MESSAGE != 'exit':
    tcpClientA.send(MESSAGE)     
    data = tcpClientA.recv(BUFFER_SIZE)
    print(" Client 2 received data:", data)
    MESSAGE = input("Client 2: Enter message to continue/ Enter exit:")
    MESSAGE = utils.stringToBytes(MESSAGE)

tcpClientA.close() 