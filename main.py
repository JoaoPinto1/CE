import gymnasium as gym
import math
import random
from maps_to_evaluate import *
from variation_operators import *
from selection import tournament, survivor_elitism, survivor_generational
from visuals import create_interactive_plot
from evolutionary_algorithm import *

map = 8
if map == 4:
    env = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)
if map == 8:
    env = gym.make('FrozenLake-v1', desc=map_8_by_8, is_slippery=False)
if map == 12:
    env = gym.make('FrozenLake-v1', desc=map_12_by_12, is_slippery=False)

def function_fitness(config):
    def fitness(individual):
        pheno = config['mapping'](individual['genotype'])
        return function_evaluation(pheno)
    return fitness

def function_evaluation(phenotype):
    #TODO: implement fitness function
    fitness = 0
    row = math.floor(phenotype[-1][0] / map)
    col = phenotype[-1][0] % map
    fitness = (map-1)-row + (map-1)-col
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
    for i in range(config['genotype_size']):
        action = random.randint(0, 3)
        genotype.append(action)
    return {'genotype': genotype, 'fitness': None}


if __name__ == '__main__':
    # Dictonary with Configurations for the Evolutionary Algorithm
    config = {
        'population_size' : 100,
        'generations' : 2000,
        'genotype_size' : MAX_ITERATIONS_4_by_4,
        'prob_crossover' : 0.9,
        'prob_mutation' : 0.1,
        'seed' : 202, #int(sys.argv[1]),
        'generate_individual' : generate_random_individuals,
        'mapping' : mapping,
        'maximization' : False,
        'mutation' : swap_mutation, #TODO: implement mutation function,
        'crossover' : uniform_crossover, #TODO: implement crossover function,
        'parent_selection' : tournament(5, maximization=False),
        'survivor_selection' : survivor_elitism(.02, maximization=False),
        'fitness_function' : None,
        'interactive_plot' : create_interactive_plot('Evolving...', 'Iteration', 'Quality', (0, 2000), (-2, 10)),
    }
    config['fitness_function'] = function_fitness(config)
    
    random.seed(config['seed'])
    observation, info = env.reset(seed=22)
    bests = ea(config)
