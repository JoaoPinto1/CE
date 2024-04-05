import gymnasium as gym
import math
import random
from maps_to_evaluate import *
from variation_operators import *
from selection import tournament, survivor_elitism, survivor_generational
from visuals import create_interactive_plot
from evolutionary_algorithm import *

env = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)

def function_fitness(config):
    def fitness(individual):
        pheno = config['mapping'](individual['genotype'])
        return function_evaluation(pheno)
    return fitness

def function_evaluation(phenotype):
    #TODO: implement fitness function
    fitness = 0
    row = math.floor(phenotype[-1][0] / 4)
    col = phenotype[-1][0] % 4
    fitness = 3-row + 3-col
    if phenotype[-1][1] and not phenotype[-1][2]:
        fitness *= 10
    fitness += len(phenotype)
    return fitness

def mapping(genotype):
    phenotype = []
    observation, info = env.reset(seed=config['seed'])
    for step in range(config['genotype_size']):
        action = genotype[step]
        observation, reward, terminated, truncated, info = env.step(action)
        phenotype.append([observation, terminated, reward])
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
        'population_size' : 15,
        'generations' : 1000,
        'genotype_size' : MAX_ITERATIONS_4_by_4,
        'prob_crossover' : 0.90,
        'prob_mutation' : 0.1,
        'seed' : 2024, #int(sys.argv[1]),
        'generate_individual' : generate_random_individuals,
        'mapping' : mapping,
        'maximization' : False,
        'mutation' : swap_mutation, #TODO: implement mutation function,
        'crossover' : two_point_crossover, #TODO: implement crossover function,
        'parent_selection' : tournament(5, maximization=False),
        'survivor_selection' : survivor_elitism(.02, maximization=False),
        'fitness_function' : None,
        'interactive_plot' : create_interactive_plot('Evolving...', 'Iteration', 'Quality', (0, 2000), (-2, 10)),
    }
    config['fitness_function'] = function_fitness(config)
    
    random.seed(config['seed'])
    observation, info = env.reset(seed=config['seed'])
    bests = ea(config)
