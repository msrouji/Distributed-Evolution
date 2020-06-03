# Distributed Evolution Experiments

### Usage

First, run `python client.py` in a few terminals. These clients will be listening for a server...you can add as many clients as you like.

Next, run `python server_sync.py` or `python server_async.py` in a new terminal for synchronous or asynchronous updates, respectively. The server will then accept connections from clients.

The default update mode for server_sync is to take the maximum-fitness weightset of the previous iteration. To instead use a linear combination based on the fitness scores (scaling such that the coefficients add to 1), change the target of `Thread(target=on_new_client_max,args=(conn,addr)).start()` in the while(True) loop at the bottom from `on_new_client_max` to `on_new_client_lincomb`.