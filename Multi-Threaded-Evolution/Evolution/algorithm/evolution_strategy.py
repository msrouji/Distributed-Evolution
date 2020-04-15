import numpy as np
import multiprocessing as mp
import time 

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle

class EvolutionStrategy(object):
    def __init__(self, 
                 weights, 
                 get_reward_func, 
                 population_size=100, 
                 sigma=0.1, 
                 learning_rate=0.03, 
                 decay=0.999,
                 num_threads=20):

        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads

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

    def _worker_process(self, args):
        get_reward, noise, weights = args
        w0 = self._get_weights_try(weights, noise, negate=False)
        w1 = self._get_weights_try(weights, noise, negate=True)
        return get_reward(w0), get_reward(w1), noise

    def _get_weights_try(self, w, p, negate=False):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            if negate:
                weights_try.append(w[index] - jittered)
            else:
                weights_try.append(w[index] + jittered)

        return weights_try

    def _get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                noise = np.random.randn(*w.shape)
                #noise = np.random.zipf(2.0, w.shape) / 100.0
                #noise = np.random.uniform(-1.0, 1.0, w.shape)
                x.append(noise)
            population.append(x)
        return population

    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = iter((self.get_reward, p, self.weights) for p in population)
            rewards = pool.imap_unordered(self._worker_process, worker_args)
        else:
            rewards = iter((self.get_reward(self._get_weights_try(self.weights, p, negate=False)), 
                            self.get_reward(self._get_weights_try(self.weights, p, negate=True)), 
                            p) for p in population)
        
        return rewards

    def _update_weights(self, rewards, population, wait):
        stats = []
        cache = []
        
        mean = None
        std = None
        count = 0
        base_reward = self.get_reward(self.weights)
        for r0, r1, noise in rewards:
            stats.append(r0)
            #stats.append((r0-r1)/2)
            cache.append(noise)
            count += 1
            
            if count >= wait*self.POPULATION_SIZE:
                stats = np.array(stats)
                mean = stats.mean()
                std = stats.std()
                
                update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)

                for i in range(int(wait*self.POPULATION_SIZE)):
                    norm = (stats[i] - mean) / (std + 1e-16)
                    #norm = stats[i]
                    for j in range(len(self.weights)):
                        self.weights[j] = self.weights[j] + update_factor * norm * cache[i][j]

                break

        self.learning_rate *= self.decay

    def _startup_message(self, name, wait, avg_runs, print_step, save_step):
        print("\n------------------------------------------------")
        print("Launching Evolutionary Strategies (ES) Algorithm")
        print("------------------------------------------------\n")
        print("Overview")
        print("--------")
        print("Total Number of Trainable Parameters: %d" % (1))
        print("Shape of Weights (Parameters): "+str(1))
        print("--------")
        print("Population Size: %d" % (self.POPULATION_SIZE))
        print("Number of CPU-Threads: %d" % (self.num_threads))
        print("Wait Percentage: %f" % (wait))
        print("--------")
        print("Noise Standard Deviation: %f" % (self.SIGMA))
        print("Learning Rate: %f" % (self.learning_rate))
        print("Learning Rate Decay: %f" % (self.decay))
        print("--------")
        print("Average Accross (episodes): %d" % (avg_runs))
        print("Print Each (iterations) " + str(print_step))
        print("Save Weights Each (iterations) " + str(save_step))
        print("--------\n")
        print("Environment Name: "+name+"\n")


    def run(self, iterations, name="", wait=0.8, avg_runs=5, print_step=10, save_step=10):
        start = time.time()
        
        self._startup_message(name, wait, avg_runs, print_step, save_step)
        
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        for iteration in range(iterations):

            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population, wait)

            if not (print_step is None) and (iteration + 1) % print_step == 0:
                avg = sum([self.get_reward(self.weights) for _ in range(avg_runs)]) / avg_runs
                print('time %0.2fs, iter %d. reward: %0.2f' % (time.time() - start, iteration + 1, avg))

            if not (save_step is None) and (iteration + 1) % save_step == 0:
                self.save(filename="ES_weights_"+str(iteration+1)+".pkl")
        
        if pool is not None:
            pool.close()
            pool.join()
