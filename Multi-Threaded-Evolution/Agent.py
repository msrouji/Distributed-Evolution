import random
import numpy as np
from evostra import EvolutionStrategy
import gym

class Agent:

    def __init__(self, 
                 env, 
                 seed, 
                 model, 
                 layers, 
                 population_size=20, 
                 sigma=0.1, 
                 learning_rate=0.03, 
                 decay=0.999, 
                 num_threads=20, 
                 eps_avg=1,
                 wait=0.8,
                 avg_runs=5,
                 print_step=1,
                 save_step=None)

        self.env = gym.make(env)
        
        self.env.seed(seed)
        np.random.seed(seed)

        self.observation_shape = self.env.observation_space.shape
        if "Discrete" in str(self.env.action_space):
            self.action_shape = (self.env.action_space.n,)
            self.discrete = True
        else:
            self.action_shape = self.env.action_space.shape
            self.discrete = False

        self.model = model([self.observation_shape[0]]+layers+[self.action_shape[0]], self.discrete)

        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = num_threads

        self.eps_avg = eps_avg
        self.wait = wait
        self.avg_runs = avg_runs
        self.print_step = print_step
        self.save_step = save_step

        self.es = EvolutionStrategy(self.model.get_weights(), 
                                    self.get_reward, 
                                    self.population_size, 
                                    self.sigma, 
                                    self.learning_rate, 
                                    self.decay,
                                    self.num_threads)

    def get_weights(self):
        return self.es.weights

    def get_predicted_action(self, observation):
        prediction = self.model.predict(observation)
        return prediction

    def train(self, iterations):
        self.es.run(iterations, 
                    wait=self.wait, 
                    avg_runs=self.avg_runs, 
                    print_step=self.print_step, 
                    save_step=self.save_step)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in range(self.eps_avg):
            # reset the environment and get first observation
            observation = None
            # stopping criteria
            done = False
            while not done:
                # get action from model
                action = self.get_predicted_action(observation)
                # apply action to environment and receive next observation, reward, and done
                reward, done, next_observation = None, None, None

                observation = next_observation
                total_reward += reward

        return total_reward / self.eps_avg

    def play_episodes(self, num_eps, render=True):
        total_reward = 0.0
        self.model.set_weights(self.get_weights())

        for episode in range(num_eps):
            pass

        print("Average reward across "+str(num_eps)+" trials is "+str(total_reward / num_eps))





