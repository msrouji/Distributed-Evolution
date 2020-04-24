from OpenAIGym import *
#from Stego import *
from evostra import FeedForwardNetwork
from evostra import CNN2D

import numpy as np

"""
model = CNN2D((2, 10, 10), (1,), 
              [{"type": "conv2d", "kernel":(32, 3, 3), "padding": (1, 1), "stride": (1, 1), "activation": ("tanh", (None,))}, 
               {"type": "conv2d", "kernel":(1, 3, 3), "padding": (1, 1), "stride": (1, 1), "activation": ("column-softmax", (None,))}])

               #{"type": "dense", "size": 4, "activation": ("column-softmax", (None,))}])

print(model.get_parameter_count())

inp = np.ones((2, 10, 10))
out = model.predict(inp)

print(out.shape)
print(np.sum(out))
print(out[0].sum(0))
"""
"""
agent = Agent(3,
              population_size=100,
              sigma=0.15,
              learning_rate=0.02,
              decay=0.999,
              num_threads=2,
              eps_avg=1,
              wait=1.0,
              avg_runs=1,
              print_step=1,
              save_step=None)

agent.train(5000)
"""

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
agent.train(500)

agent.play_episodes(2)

