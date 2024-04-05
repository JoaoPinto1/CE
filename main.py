import gymnasium as gym
import random
from maps_to_evaluate import *
from variation_operators import order_crossover, swap_mutation, cycle_crossover, scramble_mutation, uniform_crossover
from selection import tournament, survivor_elitism, survivor_generational
from visuals import create_interactive_plot
from evolutionary_algorithm import *

env = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)
observation, info = env.reset(seed=42)

def function_fitness(config):
    def fitness(individual):
        pheno = config['mapping'](individual['genotype'])
        return function_evaluation(pheno)
    return fitness

def function_evaluation(phenotype):
    #TODO: implement fitness function
    fitness = phenotype[-1]
    return fitness

def mapping(genotype):
    phenotype = []
    for step in range(config['genotype_size']):
        action = genotype[step]
        observation, reward, terminated, truncated, info = env.step(action)
        phenotype.append(observation)
        if terminated:
            break
    return phenotype

def generate_random_individuals(config):
    genotype = []
    #TODO: implement function to create random individuals
    last = None
    for i in range(config['genotype_size']):
        action = random.randint(0, 3)
        #while action == 0 and last == 2 or action == 2 and last == 0 or action == 1 and last == 3 or action == 3 and last == 1:
            #action = random.randint(0, 3)
        genotype.append(action)
        last = action
    return {'genotype': genotype, 'fitness': None}


if __name__ == '__main__':
    # Dictonary with Configurations for the Evolutionary Algorithm
    config = {
        'population_size' : 100,
        'generations' : 2000,
        'genotype_size' : MAX_ITERATIONS_4_by_4,
        'prob_crossover' : 0.9,
        'prob_mutation' : 0.05,
        'seed' : 2024, #int(sys.argv[1]),
        'generate_individual' : generate_random_individuals,
        'mapping' : mapping,
        'maximization' : True,
        'mutation' : swap_mutation, #TODO: implement mutation function,
        'crossover' : uniform_crossover, #TODO: implement crossover function,
        'parent_selection' : tournament(5, maximization=True),
        'survivor_selection' : survivor_elitism(.02, maximization=True),
        'fitness_function' : None,
        'interactive_plot' : create_interactive_plot('Evolving...', 'Iteration', 'Quality', (0, 2000), (-2, 10)),
    }
    config['fitness_function'] = function_fitness(config)
    
    random.seed(config['seed'])
    bests = ea(config)
