#Evolutionary Algorithm for the Frozen Lake Problem
#Author: André Moreira, João Pinto

import random


## Crossovers ##

def two_point_crossover(parent1, parent2):
    cut_points1 = sorted(random.sample(range(len(parent1['genotype'])), 2))
    cut_points2 = sorted(random.sample(range(len(parent2['genotype'])), 2))
    genotype = parent1['genotype'][: cut_points1[0] ] +  parent2['genotype'][ cut_points2[0] : cut_points2[1]] + parent1['genotype'][ cut_points1[1] : ]
    return {'genotype': genotype, 'fitness': None}

def one_point_crossover(parent1, parent2):
    size_1 = len(parent1['genotype'])
    size_2 = len(parent2['genotype'])    
    pos_1 = random.randint(1, size_1 - 1)
    pos_2 = random.randint(1, size_2 - 1)
    genotype = parent1['genotype'][:pos_1] + parent2['genotype'][pos_2:]
    return {'genotype': genotype, 'fitness': None}


## Mutations ##


def delete_mutation(p):
    p['fitness'] = None
    at = random.randint(0, len(p['genotype']) - 1)
    p['genotype'].pop(at)
    return p

def change_value_mutation(p):
    p['fitness'] = None
    at = random.randint(0, len(p['genotype']) - 1)
    p['genotype'][at] = random.randint(0, 3) 
 
    return p

def insert_mutation(p):
    p['fitness'] = None
    action = random.randint(0, 3)
    at = random.randint(0, len(p['genotype']) - 1)
    p['genotype'].insert(at, action)
    return p

def main_mutation(p):
    op = random.randint(1, 3)
    if op == 1:
        p = delete_mutation(p)
    elif op == 2:
        p = change_value_mutation(p)
    else:
        p = insert_mutation(p)
    return p
    



