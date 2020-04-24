import numpy as np
import random

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class FeedForwardNetwork(object):
    def __init__(self, layer_sizes, discrete):
        self.weights = []
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.zeros(shape=(layer_sizes[index], layer_sizes[index+1])))

        self.discrete = discrete
        
        self.iteration = 0

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for layer in self.weights[:-1]:
            out = np.dot(out, layer)
            out = np.tanh(out)

        out = np.dot(out, self.weights[-1])
        if self.discrete:
            if True: #self.iteration > 5000:
                return np.argmax(out)
            
            out = out[0]
            out = list(enumerate(out))
            out.sort(reverse=True, key=lambda t: t[1])
            
            ranked = list(map(lambda t: t[0], out))
            start = 1000
            count = start
            bag = []
            for i in range(len(ranked)):
                action = ranked[i]
                bag = bag + count*[action]
                count = start // (i + 2)
            
            #random.shuffle(bag)
            
            self.iteration += 1
            return random.choice(bag)
            

        else:
            out = np.tanh(out)
            return out[0]

    def get_weights(self):
        return self.weights

    def get_parameter_count(self):
        total = 0
        for w in self.weights:
            total += np.prod(w.shape)

        return total

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
