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
    it = 0
    #Evaluate how good the individuals are (problem dependent)
    for it in range(it, config['generations']):
        for i in population:
            if i['fitness'] == None:
                evaluate(i, config)
        population.sort(key = lambda x: x['fitness'], reverse=config['maximization'])
        best = (config['mapping'](population[0]), population[0]['fitness'], len(population[0]['steps']))
        #if config['interactive_plot'] is not None:
           #update_graph(it, best[1], *config['interactive_plot'])
        print("Gen:", it, best[0], best[1], best[2])
        bests.append(best)
        new_population = []
        while len(new_population) < config['population_size']:
            if random.random() < config['prob_crossover']:
                #Parent Selection
                p1 = config['parent_selection'](population)
                p2 = config['parent_selection'](population)
                #Recombination
                ni = config['crossover'](p1, p2)
                ni = run_env(config, ni['genotype'])
            else:
                ni = config['parent_selection'](population)
                ni = run_env(config, ni['genotype'])
            #Mutation 
            if random.random() < config['prob_mutation']:
                ni = config['mutation'](ni)
                ni = run_env(config, ni['genotype'])
            new_population.append(ni)
        population = config['survivor_selection'](population, new_population)
    return bests

def run_env(config, genotype):
    steps = []
    observation, info = config['env'].reset(seed=config['seed'])
    for step in range(len(genotype)):
        action = genotype[step]
        observation, reward, terminated, truncated, info = config['env'].step(action)
        steps.append([observation, terminated, reward])
        if terminated:
            #genotype = genotype[:step + 1]
            break
    return {'genotype': genotype, 'steps': steps, 'fitness': None}
