import gymnasium as gym
import math
import random
from maps_to_evaluate import *
from variation_operators import *
from selection import tournament, survivor_elitism, survivor_generational
from visuals import create_interactive_plot
from evolutionary_algorithm import *


def function_fitness(config):
    def fitness(individual):
        fitness = function_evaluation(config['mapping'](individual))
        return fitness
    return fitness

def function_evaluation(steps):
    fitness = 0
    map = config['map_size']

    row = math.floor(steps[-1][0] / map)
    col = steps[-1][0] % map
    
    #Manhattan distance to the goal
    fitness += (map-1)-row + (map-1)-col
    
    #if the last step is a hole or didn't finish
    if (steps[-1][1] and not steps[-1][2]) or not steps[-1][1]:
        fitness *= 100

    #Number of steps
    fitness += len(steps)

    #if there are repeating positions
    positions = [sublist[0] for sublist in steps]
    if len(positions) != len(set(positions)):
        fitness *= 10

    return fitness

def mapping(genotype):
    return genotype['steps']

def generate_random_individuals(config):
    genotype = []
    for i in range(config['genotype_size']):
        action = random.randint(0, 3)
        genotype.append(action)
    return run_env(config, genotype)

if __name__ == '__main__':
    # Dictonary with Configurations for the Evolutionary Algorithm
    config = {
        'map_size' : 4,
        'runs' : 3,
        'population_size' : 100,
        'generations' : 200,
        'prob_crossover' : 0.9,
        'prob_mutation' : 0.1,
        'seed' : 2024, #int(sys.argv[1]),
        'generate_individual' : generate_random_individuals,
        'mapping' : mapping,
        'maximization' : False,
        'mutation' : change_value_mutation, #TODO: implement mutation function,
        'crossover' : one_point_crossover, #TODO: implement crossover function,
        'parent_selection' : tournament(5, maximization=False),
        'survivor_selection' : survivor_elitism(.02, maximization=False),
        'fitness_function' : None,
        'interactive_plot' : create_interactive_plot('Evolving...', 'Iteration', 'Quality', (0, 2000), (-2, 10)),
    }

    if config['map_size'] == 4:
        config['env'] = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)
        config['genotype_size'] = MAX_ITERATIONS_4_by_4

    elif config['map_size'] == 8:
        config['env'] = gym.make('FrozenLake-v1', desc=map_8_by_8, is_slippery=False)
        config['genotype_size'] = MAX_ITERATIONS_8_by_8

    else:
        config['env'] = gym.make('FrozenLake-v1', desc=map_12_by_12, is_slippery=False)
        config['genotype_size'] = MAX_ITERATIONS_12_by_12

    config['fitness_function'] = function_fitness(config)
    
    best_overall = []
    average_overall = []
    
    observation, info = config['env'].reset(seed=config['seed'])
    
    for i in range(config['runs']):
        config['seed'] += 1
        best , average = ea(config)
        
        if best_overall == []:
            best_overall = best
            average_overall = average
            
        elif best_overall[-1][1] > best[-1][1]:
            best_overall = best
            average_overall = average     
    
    # write results to file
    if map == 4:
        with open('map_4_by_4.txt', 'w') as file:
            for item in best_overall:
                file.write("%s\n" % str(item))
        with open('map_4_by_4_average.txt', 'w') as file:
            for item in average_overall:
                file.write("%s\n" % str(item))
                
    elif map == 8:
        with open('map_8_by_8.txt', 'w') as file:
            for item in best_overall:
                file.write("%s\n" % str(item))
        with open('map_8_by_8_average.txt', 'w') as file:
            for item in average_overall:
                file.write("%s\n" % str(item))
                
    else:
        with open('map_12_by_12.txt', 'w') as file:
            for item in best_overall:
                file.write("%s\n" % str(item))
        with open('map_12_by_12_average.txt', 'w') as file:
            for item in average_overall:
                file.write("%s\n" % str(item))
    