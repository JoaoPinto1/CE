#Evolutionary Algorithm for the Frozen Lake Problem
#Author: André Moreira, João Pinto

import gymnasium as gym
import statistics
import math
import random
from maps_to_evaluate import *
from variation_operators import *
from selection import tournament, survivor_elitism, survivor_generational
from visuals import create_interactive_plot
from evolutionary_algorithm import *
from scipy import stats
import matplotlib.pyplot as plt 
import numpy as np
from itertools import product

def get_data(data1, data2, average1, pop1 , pmut1 , pcross1 , mut1 , cross1, elitism1, average_first_reached, std):
    normal1 = test_normal_ks(data1)
    normal2 = test_normal_ks(data2)
    print(data1)
    print(data2)
    if(normal1 and normal2):
        if(ttest(data1,data2)):
            write_stats_file(pcross1 , pmut1 , pop1 ,mut1 , cross1 , average1 , average_first_reached, elitism1,std)
            with open("statistics.txt", 'a') as file:
                file.write("T-test - There is a statistically significant difference in average fitness compared to previous runs.\n")
                file.write("#######################################################\n\n")
        else:
            write_stats_file(pcross1 , pmut1 , pop1 ,mut1 , cross1 , average1 , average_first_reached, elitism1,std)
            with open("statistics.txt", 'a') as file:
                file.write("T-test - There is no statistically significant difference in average fitness compared to previous runs.\n")
                file.write("#######################################################\n\n")
    else:
        if(wilcoxon(data1,data2)):
            write_stats_file(pcross1 , pmut1 , pop1 ,mut1 , cross1 , average1 , average_first_reached, elitism1,std)
            with open("statistics.txt", 'a') as file:
                file.write("Wilcoxon - There is a statistically significant difference in average fitness compared to previous runs.\n")
                file.write("#######################################################\n\n")
        else:
            write_stats_file(pcross1 , pmut1 , pop1 ,mut1 , cross1 , average1 , average_first_reached, elitism1,std)
            with open("statistics.txt", 'a') as file:
                file.write("Wilcoxon - There is no statistically significant difference in average fitness compared to previous runs.\n")
                file.write("#######################################################\n\n")


def write_stats_file(pcross1 , pmut1 , pop1 ,mut1 , cross1 , average1, average_first_reached, elitism1,std):
    with open("statistics.txt", 'a') as file:
        file.write("#######################################################\n\n")
        file.write("Prob. Crossover: %s\n" % pcross1)
        file.write("Prob. Mutation: %s\n" % pmut1)
        file.write("Population Size: %s\n" % pop1)
        file.write("Mutation: " + mut1.__name__ + "\n")
        file.write("Crossover: " + cross1.__name__ + "\n")
        file.write("Average obtained: %s +- %s\n" % (average1, std))
        file.write("Average first reached: %s\n" % average_first_reached)
        file.write("Elitism: %s\n" % elitism1)

def test_normal_ks(data):
    """Kolgomorov-Smirnov"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    D, p_value = stats.kstest(norm_data, 'norm')
    
    if p_value > 0.05:
        print("Data is likely normally distributed (fail to reject null hypothesis)")
        return True
    else:
        print("Data is not likely normally distributed (reject null hypothesis)")
        return False

        
def wilcoxon(data1,data2):
    U_statistic, p_value = stats.wilcoxon(data1,data2)

    if p_value < 0.05:
        print("There is a statistically significant difference between the two samples.")
        return True
    else:
        print("There is no statistically significant difference between the two samples.")
        return False

def ttest(previous_fitness , current_fitness):
    
    # Perform t-test for average fitness
    avg_t_stat, avg_p_value = stats.ttest_rel(previous_fitness, current_fitness)

    # Interpret results
    alpha = 0.05
    if avg_p_value < alpha:
        print("The first configuration is better than the second configuration with 95% confidence.")
        return True
    else:
        print("There is no statistically significant difference between the two configurations.")
        return False


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


def teste(config, pop_size, pmut, pcross, mut, cross, elitism):
    config['population_size'] = pop_size
    config['prob_mutation'] = pmut
    config['prob_crossover'] = pcross
    config['mutation'] = mut
    config['crossover'] = cross
    if elitism:
        config['survivor_selection'] = survivor_elitism(.02, maximization=False)
    else:
        config['survivor_selection'] = survivor_generational
    print("Testing configuration: ", pop_size, pmut, pcross, mut.__name__, cross.__name__, elitism)
    data2 = []
    for i in range(config['runs']):
        best_fitness = 999999999999 
        total_fitness = 0
        random.seed(config['seed'])
        observation, info = config['env'].reset(seed=config['seed'])
        best = ea(config)
        config['seed'] += 1

        for a, item in enumerate(best):
            value_at_i = item[1]  
            total_fitness += value_at_i 
            # Compare the value with the current minimum
            if value_at_i < best_fitness:
                best_fitness = value_at_i
                first_max_index = a
        
        fitness_reached.append(first_max_index)
        average_first_reached = sum(fitness_reached) / len(fitness_reached)

        data2.append(total_fitness / config['generations'])
        print("Run: ", i)

    average_fitness = sum(data2) / len(data2)
    std = np.std(data2)
    config['seed'] = 1234
    print("Average fitness: ", average_fitness, "Standard deviation: ", std)
    return [data2, average_fitness, pop_size, pmut, pcross, mut, cross, elitism, average_first_reached, std]


if __name__ == '__main__':
    # Dictonary with Configurations for the Evolutionary Algorithm
    config = {
        'map_size' : 12,
        'runs' : 1,
        'population_size' : 150,
        'generations' : 50,
        'prob_crossover' : 0.8,
        'prob_mutation' : 0.05,
        'seed' : 1234, #int(sys.argv[1]),
        'generate_individual' : generate_random_individuals,
        'mapping' : mapping,
        'maximization' : False,
        'mutation' : main_mutation, #TODO: implement mutation function,
        'crossover' : one_point_crossover, #TODO: implement crossover function,
        'parent_selection' : tournament(5, maximization=False),
        'survivor_selection' : survivor_elitism(.02, maximization=False),
        'fitness_function' : None,
        'interactive_plot' : create_interactive_plot('Evolving...', 'Iteration', 'Quality', (0, 50), (10, 200)),
    }
    
    if config['map_size'] == 4:
        config['env'] = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False)

    elif config['map_size'] == 8:
        config['env'] = gym.make('FrozenLake-v1', desc=map_8_by_8, is_slippery=False)

    else:
        config['env'] = gym.make('FrozenLake-v1', desc=map_12_by_12, is_slippery=False)
    
    config['fitness_function'] = function_fitness(config)
    
    data1 = []
    fitness_reached = []
    for i in range(config['runs']):
        best_fitness = 999999999999
        total_fitness = 0
        random.seed(config['seed'])
        observation, info = config['env'].reset(seed=config['seed'])
        best = ea(config)
        config['seed'] += 1

        for a, item in enumerate(best):
            value_at_i = item[1]  
            total_fitness += value_at_i 
            # Compare the value with the current minimum
            if value_at_i < best_fitness:
                best_fitness = value_at_i
                first_max_index = a
        
        fitness_reached.append(first_max_index)

        data1.append(total_fitness / config['generations'])
        print("Run: ", i)
    config['seed'] = 1234
    average1 = np.mean(data1)
    std = np.std(data1)
    average_first_reached = np.mean(fitness_reached)

    print("Average fitness: ", average1, "Standard deviation: ", std, "Average first reached: ", average_first_reached)
    write_stats_file(0.8 , 0.05 , 150 ,main_mutation , one_point_crossover , average1, average_first_reached, True, std)
    
    testes = []
    testes.append(teste(config, 50, 0.05, 0.8, main_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.05, 0.8, delete_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.1, 0.8, main_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.05, 0.9, main_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.05, 0.7, main_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.05, 0.8, main_mutation, two_point_crossover, True))
    testes.append(teste(config, 150, 0.05, 0.8, main_mutation, two_point_crossover, False))
    testes.append(teste(config, 150, 0.05, 0.8, main_mutation, one_point_crossover, False))
    testes.append(teste(config, 100, 0.05, 0.8, main_mutation, one_point_crossover, False))
    testes.append(teste(config, 150, 0.05, 0.8, insert_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.05, 0.8, change_value_mutation, one_point_crossover, True))
    testes.append(teste(config, 150, 0.2, 0.8, main_mutation, one_point_crossover, True))

    for test in testes:
        get_data(data1, test[0], test[1], test[2], test[3], test[4], test[5], test[6], test[7], test[8], test[9])
