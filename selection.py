import random
import copy


## Parent Selection ##
def tournament(tournament_size : int, maximization=True) -> dict:
    def tournament(population):
        pool = random.sample(population, tournament_size)
        pool.sort(key=lambda i: i['fitness'], reverse=maximization)
        return copy.deepcopy(pool[0])
    return tournament


## Survivals Selection ##
def survivor_elitism(elite : float, maximization=True):
    def elitism(parents,offspring):
        pop_size = len(parents)
        elite_size = int(pop_size * elite)
        parents.sort(key=lambda x: x['fitness'], reverse=maximization)
        new_population = parents[ : elite_size] + offspring[ : pop_size - elite_size]
        return new_population
    return elitism


def survivor_generational(parents, offspring):
    return offspring
