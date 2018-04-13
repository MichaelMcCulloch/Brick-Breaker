#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import numpy as np
import sys
import json

from Agent import Agent
from Environment import Environment

COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()
name = MPI.Get_processor_name()

config = None


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mix_pop(population):
    # gather() population into root node
    all_pop = COMM.gather(population, root=0)
    if rank == 0:
        # flatten gathered list of lists
        all_pop = [b for val in all_pop for b in val]
        np.random.shuffle(all_pop)                 # shuffle population
        # chunk it for transmission
        all_pop = list(chunks(all_pop, len(all_pop)//size))
    new_pop = COMM.scatter(all_pop, root=0)
    return new_pop


def make_population():
    '''
    Each machine generates it's own population. 
    Assume config.json is mirrored across nodes
    '''

    pop_size = config['Population_Size']
    gene_param = config['Genetic_Parameters']
    low_NHU, high_NHU = gene_param['Num_Hidden_Units_Range']
    n_Layers = gene_param['Max_Conv_Layers']
    kernel_max, stride_Max = gene_param['Layer_KS_Limits']

    initial_fitness = np.repeat(None, pop_size)
    rand_h_size = np.random.randint(low_NHU, high_NHU, pop_size)
    rand_layer_count = np.random.randint(2, n_Layers+1, pop_size)
    rand_kern = np.random.randint(2, kernel_max+1, pop_size * n_Layers)
    rand_stride = np.random.randint(2, stride_Max+1, pop_size * n_Layers)
    rand_kern = np.reshape(rand_kern, [pop_size, -1])
    rand_stride = np.reshape(rand_stride, [pop_size, -1])
    individuals = [{"Hidden_Unit_Size": h, "Layer_Count": l, "Kernel_Size": k, "Stride_Length": s}
                   for (h, l, k, s) in zip(rand_h_size, rand_layer_count, rand_kern, rand_stride)]
    return list(zip(initial_fitness, individuals))


def roulette(population):
    return np.random.choice(population)


def new_random(range, old):
    new = np.random.randint(range[0], range[1])
    if new == old:
        new = (new + 1) % range[1]
    return new


def mutate(individual):

    new_member = individual.copy()

    r = np.random.randint(0, 4)
    gene_config = config['Genetic_Parameters']
    if r == 0:
        nhu = gene_config['Num_Hidden_Units_Range']
        range = [nhu[0], nhu[1]]
        key = "Hidden_Unit_Size"
    elif r == 1:
        range = [2, gene_config['Max_Conv_Layers']]
        key = "Layer_Count"
    elif r == 2:
        range = [2, gene_config['Layer_KS_Limits'][0]]
        key = "Kernel_Size"
    else:
        range = [2, gene_config['Layer_KS_Limits'][1]]
        key = "Stride_Length"

    old_val = new_member[key]
    if r == 0 or r == 1:
        new_val = new_random(range, old_val)
    elif r == 2 or r == 3:
        t = np.random.randint(0, len(old_val))
        old_val[t] = new_random(range, old_val[t])
        new_val = old_val

    new_member.update({key: new_val})
    return new_member


def cross(A, B):

    r = np.random.randint(0, 2, 4)
    new_A = A.copy()
    new_B = B.copy()

    # bits from right to left
    keys = list()
    if (r[0]):
        keys.append('Hidden_Unit_Size')
    if (r[1]):
        keys.append('Layer_Count')
    if (r[2]):
        keys.append('Kernel_Size')
    if (r[3]):
        keys.append('Stride_Length')

    for key in keys:
        temp = new_B[key]
        new_B.update({key: new_A[key]})
        new_A.update({key: temp})

    return new_A, new_B


def breed(gene_pool, target_size, p_mutate=0.8):
    np.random.shuffle(gene_pool)
    return [(None, h[1]) for h in gene_pool][0:target_size]


if __name__ == '__main__':

    config = json.load(open("config.json"))
    env = Environment(config)
    population = make_population()
    print(population[1])

    for gen in range(config['Generations']):
        # Occasionally mix population among workers
        if gen != 0 and gen % config['Mix_Interval'] == 0:
            population = mix_pop(population)

        # generate a list of candidates
        candidates = list()
        for individual in population:
            fitness = individual[0]
            traits = individual[1]

            # don't evaluate an individual more than once
            if fitness != None:
                candidates.append(individual)
                continue

            # creating an agent may fail if parameters suck!
            agent = Agent(traits, env)
            agent.train(episode_count=config['Short_Train'])
            fitness = agent.evaluate()
            candidates.append((fitness, traits))

        population = breed(candidates, len(population),
                           p_mutate=config['Mutation_Prob'])

    '''
    Here, select the best individual, and begin distributed training
    '''
