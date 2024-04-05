import random

from visuals import update_graph


def generate_initial_population(config):
    for i in range(config['population_size']):
        yield config['generate_individual'](config)


def evaluate(ind, config):
    ind['fitness'] = config['fitness_function'](ind)


# Simple Evolutionary Algorithm
def ea(config):
    bests = []
    # Create a initial population randomly
    population = list(generate_initial_population(config))
    print(population)
    it = 0
    #Evaluate how good the individuals are (problem dependent)
    for it in range(it, config['generations']):
        for i in population:
            if i['fitness'] == None:
                evaluate(i, config)
        population.sort(key = lambda x: x['fitness'], reverse=config['maximization'])
        best = (config['mapping'](population[0]['genotype']), population[0]['fitness'])
        #if config['interactive_plot'] is not None:
           #update_graph(it, best[1], *config['interactive_plot'])
        print("Gen:", it, best[0], best[1])
        bests.append(best)
        new_population = []
        while len(new_population) < config['population_size']:
            if random.random() < config['prob_crossover']:
                #Parent Selection
                p1 = config['parent_selection'](population)
                p2 = config['parent_selection'](population)
                #Recombination
                ni = config['crossover'](p1, p2)
            else:
                ni = config['parent_selection'](population)
            #Mutation 
            if random.random() < config['prob_mutation']:
                ni = config['mutation'](ni)
            new_population.append(ni)
        population = config['survivor_selection'](population, new_population)
    return bests

