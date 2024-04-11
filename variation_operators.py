import random


## Crossovers ##

## Binary ##
def one_point_crossover(parent1, parent2):
    """
    One Point Crossover
    """
    at = random.randint(0, len(parent1['genotype'])-1)
    genotype = []
    for i in range(len(parent2['genotype'])):
        if i < at:
            genotype.append(parent1['genotype'][i])
        else:
            genotype.append(parent2['genotype'][i])
    return {'genotype': genotype, 'fitness': None} 


def two_point_crossover(parent1, parent2):
    """
    Two Point Crossover
    """
    cut_points = sorted(random.sample(range(len(parent1['genotype'])), 2))
    genotype = parent1['genotype'][ 0 : cut_points[0] ] +  parent2['genotype'][ cut_points[0] : cut_points[1]] + parent1['genotype'][ cut_points[1] : ] 
    return {'genotype': genotype, 'fitness': None}


def uniform_crossover(parent1, parent2):
    """
    Uniform Crossover
    """
    mask = [random.random() for i in range(len(parent1['genotype']))]
    genotype = []
    for i in range(len(parent1['genotype'])):
        if mask[i] < 0.5:
            genotype.append(parent1['genotype'][i])
        else:
            genotype.append(parent2['genotype'][i])
    return {'genotype': genotype, 'fitness': None} 


## Integers ##
def sample_crossover(parent1, parent2):
    size_1 = len(parent1['genotype'])
    size_2 = len(parent2['genotype'])    
    pos_1 = random.randint(0, size_1 - 1)
    pos_2 = random.randint(0, size_2 - 1)
    genotype = list(set(random.sample(parent1['genotype'], pos_1) + random.sample(parent2['genotype'], (size_2 - pos_2))))
    return {'genotype': genotype, 'fitness': None}

## Float ##
# Variation Operators : Aritmetical  Crossover
def aritmetical_cross(alpha):
    def cross(p1, p2):
        size = len(p1['genotype'])
        genotype = [None] * size
        at = random.randint(0, len(p1['genotype']) - 1)
        genotype[ : at] = p1['genotype'][ : at]
        for i in range(at, size):
            genotype[i] = alpha * p1['genotype'][i] + (1 - alpha) * p2['genotype'][i]
        return {'genotype': genotype, 'fitness': None}
    return cross


def heuristic_cross(alpha, domain):
    def cross(p1, p2):
        size = len(p1['genotype'])
        genotype = []
        # TODO: YOUR CODE HERE
        for i in range(size):
            if random.random() < alpha:
                genotype.append(p1['genotype'][i])
            else:
                genotype.append(p2['genotype'][i])
        return {'genotype': genotype, 'fitness': None}
    
    return cross

## Permutations ##
def order_crossover(p1, p2):
    size = len(p1['genotype'])
    # Select a random subsequence from parent1
    start, end = sorted(random.sample(range(size), 2))
    
    # Copy the subsequence to the offspring
    genotype = [None] * size
    genotype[start:end+1] = p1['genotype'][start : end+1]
    
    # Fill the remaining positions with elements from parent2
    # Skip the elements already present in the offspring
    p2_index = (end + 1) % size  # Start from the end of the selected subsequence in parent2
    for i in range(size - (end - start + 1)):
        while p2['genotype'][p2_index] in genotype:
            p2_index = (p2_index + 1) % size
        genotype[(end + 1 + i) % size] = p2['genotype'][p2_index]
    return {'genotype': genotype, 'fitness': None}


def cycle_crossover(p1, p2):
    size = len(p1['genotype'])
    genotype = []
   # TODO: YOUR CODE HERE
    
    return {'genotype': genotype, 'fitness': None}

## Mutations ##
## Binary ##
def bit_flip_mutation(p):
    p['fitness'] = None
    at = random.randint(0, len(p['genotype']) - 1)
    p['genotype'][at] ^= 1 
    return p

def change_value_mutation(p):
    p['fitness'] = None
    at = random.randint(0, len(p['genotype']) - 1)
    p['genotype'][at] = random.randint(0, 3) 
    return p

## Integers ##
def append_mutation(genotype_size):
    def mutation(p):
        p['fitness'] = None
        universe = set(list(range(1, genotype_size + 1)))
        elements = set(p['genotype'])
        candidates = list(universe.difference(elements))
        element = random.choice(candidates)
        p['genotype'].append(element)
        return p
    return mutation

def delete_mutation(p):
    p['fitness'] = None
    at = random.randint(0, len(p['genotype']) - 1)
    p['genotype'].pop(at)
    return p


def changing_value_mutation(neighborhood_size, genotype_size):
    def mutation(p):
        p['fitness'] = None
        index = random.choice(list(range(len(p['genotype']))))
        number = p['genotype'].pop(index)
        delta = random.choice(list(range(-neighborhood_size,neighborhood_size)))
        number = (number + delta)
        if number < 1:
            number =  1 
        if number > genotype_size:
            number = genotype_size 
        if number not in p['genotype']:
            p['genotype'].insert(index,number)
        return p
    return mutation

## Floats ##

def gaussian_mutation(sigma, domain):
    def mutation(p):
        p['fitness'] = None
        at = random.randint(0, len(p['genotype']) - 1)
        gene = random.gauss(p['genotype'][at], sigma)
        p['genotype'][at] = max(domain[0], min(gene, domain[1])) #clip to domain
        return p
    return mutation

def uniform_mutation(domain):
    def mutation(p):
        p['fitness'] = None
        at = random.randint(0, len(p['genotype']) - 1)
        p['genotype'][at] = random.uniform(domain[0], domain[1])
        return p
    return mutation

## Permutations ##

def swap_mutation(p):
    p['fitness'] = None
    index1 = random.randrange(len(p['genotype']))
    index2 = index1
    while index2 == index1:
        index2 = random.randrange(len(p['genotype']))
    p['genotype'][index1], p['genotype'][index2] = p['genotype'][index2], p['genotype'][index1]
    return p



def scramble_mutation(p):
    p['fitness'] = None
    # TODO: YOUR CODE HERE
    
    return p


