import numpy as np

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class LinearControlNetwork(object):
    def __init__(self, layer_sizes, discrete):
        self.weights = [np.zeros(shape=(layer_sizes[0], layer_sizes[1]))]
        self.discrete = discrete

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = np.dot(out, self.weights[0])

        if self.discrete:
            out = np.argmax(out)
            return out
        else:
            out = np.tanh(out)
            return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
