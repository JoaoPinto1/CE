import gymnasium as gym
import statistics
import math
import random
from maps_to_evaluate import *
from variation_operators import *
from selection import tournament, survivor_elitism, survivor_generational
from visuals import create_interactive_plot
from evolutionary_algorithm import *


def write_to_file(file_name , config):
    with open(file_name, 'a') as file:
        file.write("#######################################################\n\n")
        file.write("Prob. Crossover: %s\n" % config['prob_crossover'])
        file.write("Prob. Mutation: %s\n" % config['prob_mutation'])
        file.write("Population Size: %s\n" % config['population_size'])
        file.write("Generations: %s\n" % config['generations'])
        file.write("Mutation: " + config['mutation'].__name__ + "\n")
        file.write("Crossover: " + config['crossover'].__name__ + "\n")
        file.write("Average Fitness: %.2f\n" % average_fitness)
        file.write("Average first generation to reach best fitness: %.2f +- %.2f\n" % (average_first_reached, fitness_reached_SEM))
        file.write("Best Fitness: %s\n" % best_fitness)
        file.write("Best Length: %s\n\n" % best_length)
        file.write("#######################################################\n\n")

def function_fitness(config):
    def fitness(individual):
        fitness = simple_fitness(config['mapping'](individual))
        return fitness
    return fitness

def function_evaluation(steps):
    fitness = 0
      
    map = config['map_size']

    row = math.floor(steps[-1][0] / map)
    col = steps[-1][0] % map
    
    #Penalty early termination
    min_steps = 2*(map-1)
    if len(steps) < min_steps:
        fitness -= 1000000

    #Manhattan distance to the goal
    fitness += 10000/(1+ abs(map-1)-row + abs(map-1)-col)
    
    #if the last step is a hole or didn't finish
    if (steps[-1][1] and not steps[-1][2]) or not steps[-1][1]:
        fitness -= 100000

    if steps[-1][2]:
        fitness += 100000

    fitness += 10000/(1+len(steps))

    #if there are repeating positions
    positions = [sublist[0] for sublist in steps]
    positions.append(0) #initial position
    if len(positions) != len(set(positions)):
       fitness -= 10000 * (len(positions) - len(set(positions)))
    return fitness

def simple_fitness(steps):

    map = config['map_size']
    row = math.floor(steps[-1][0] / map)
    col = steps[-1][0] % map
    dist_to_goal = abs(map-1)-row + abs(map-1)-col

    positions = [sublist[0] for sublist in steps]
    positions.append(0) #initial position

    min_steps = 2*(map-1)

    return dist_to_goal**2 + len(steps) + (not steps[-1][1] or (steps[-1][1] and not steps[-1][2])) * map + (len(positions) - len(set(positions)))**2  #+ ((len(steps) < min_steps)*(len(steps)-min_steps))**2

def mapping(genotype):
    return genotype['steps']

def generate_random_individuals(config):
    genotype = []
    # choose random size between min_steps (2*(map-1)) and max_steps (map**2 - 1)
    genotype_size = random.randint(2*(config['map_size']-1), config['map_size']**2 - 1)

    for i in range(genotype_size):
        action = random.randint(0, 3)
        genotype.append(action)

    return run_env(config, genotype)

if __name__ == '__main__':
    # Dictonary with Configurations for the Evolutionary Algorithm
    config = {
        'map_size' : 12,
        'runs' : 3,
        'population_size' : 150,
        'generations' : 50,
        'prob_crossover' : 0.8,
        'prob_mutation' : 0.05,
        'seed' : 1234, #int(sys.argv[1]),
        'generate_individual' : generate_random_individuals,
        'mapping' : mapping,
        'maximization' : False,
        'mutation' : main_mutation, #TODO: implement mutation function,
        'crossover' : sample_crossover, #TODO: implement crossover function,
        'parent_selection' : tournament(5, maximization=False),
        'survivor_selection' : survivor_elitism(.02, maximization=False),
        'fitness_function' : None,
        'interactive_plot' : create_interactive_plot('Evolving...', 'Iteration', 'Quality', (0, 2000), (-2, 10)),
    }
    
    if config['map_size'] == 4:
        config['env'] = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)

    elif config['map_size'] == 8:
        config['env'] = gym.make('FrozenLake-v1', desc=map_8_by_8, is_slippery=False)

    else:
        config['env'] = gym.make('FrozenLake-v1', desc=map_12_by_12, is_slippery=False)
    
    config['fitness_function'] = function_fitness(config)
    
    best_overall = []
    average_overall = []
    total_fitness = 0
    first_max_index = 0
    fitness_reached = []


    for map in range(3):
        config['map_size'] = 4 + map * 4
        if config['map_size'] == 4:
            config['env'] = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)

        elif config['map_size'] == 8:
            config['env'] = gym.make('FrozenLake-v1', desc=map_8_by_8, is_slippery=False)

        else:
            config['env'] = gym.make('FrozenLake-v1', desc=map_12_by_12, is_slippery=False)

        for k in range(12):
            config['seed'] = 1234

            if k == 0:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 1:
                config['population_size'] = 100
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 2:
                config['population_size'] = 50
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 3:
                config['population_size'] = 150
                config['prob_mutation'] = 0.1
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 4:
                config['population_size'] = 150
                config['prob_mutation'] = 0.075
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 5:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.7
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 6:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.9
                config['mutation'] = main_mutation
                config['crossover'] = sample_crossover
            if k == 7:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = delete_mutation
                config['crossover'] = sample_crossover
            if k == 8:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = insert_mutation
                config['crossover'] = sample_crossover
            if k == 9:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = change_value_mutation
                config['crossover'] = sample_crossover
            if k == 10:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = one_point_crossover
            if k == 11:
                config['population_size'] = 150
                config['prob_mutation'] = 0.05
                config['prob_crossover'] = 0.8
                config['mutation'] = main_mutation
                config['crossover'] = two_point_crossover
            
            for i in range(config['runs']):
                best_fitness = -999999999999
                random.seed(config['seed'])
                observation, info = config['env'].reset(seed=config['seed'])
                best = ea(config)
                config['seed'] += 1

                # Calculate the average fitness , the first time the fitness is reached
                for i, item in enumerate(best):
                    value_at_i = item[1]  
                    total_fitness += value_at_i  # Add the value to the total
                    
                    # Compare the value with the current minimum
                    if value_at_i < best_fitness:
                        best_fitness = value_at_i
                        first_max_index = i

                fitness_reached.append(first_max_index)
                
                if best_overall == []:
                    best_overall = best
                    
                elif best_overall[-1][1] > best[-1][1]:
                    best_overall = best  
            
            average_first_reached = sum(fitness_reached) / len(fitness_reached)
            average_fitness = total_fitness / (config['runs'] * config['generations'])
            best_fitness = best_overall[-1][1]
            best_length = len(best_overall[-1][0])

            # Calculate Standard Error of the Mean (SEM)
            mean_value = statistics.mean(fitness_reached)
            std_dev = statistics.stdev(fitness_reached)
            n = len(fitness_reached)
            
            fitness_reached_SEM = std_dev / math.sqrt(n)

            # write results to file
            if config['map_size'] == 4:
                write_to_file('map_4_by_4.txt', config)
            elif config['map_size'] == 8:
                write_to_file('map_8_by_8.txt', config)
            else:
                write_to_file('map_12_by_12.txt', config)
