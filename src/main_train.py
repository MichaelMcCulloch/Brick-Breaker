#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
from random import shuffle
import numpy as np
import sys
import json

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
    '''
        gather all population into root node
        shuffle population
        scatter population
    '''
    #make transmittable dict
    dict_pop = population
    #transmit
    all_pop = COMM.gather(dict_pop, root = 0)
    if rank == 0:
        #remove dict status and shuffle
        all_pop = [b for val in all_pop for b in val]
        shuffle(all_pop)
        all_pop = list(chunks(all_pop, len(all_pop)//size))
    new_pop = COMM.scatter(all_pop, root = 0)
    return new_pop

def evaluate(population):
    
    fitness = list()
    for individual in population:
        '''
        Create an agent using the parameters of this individual
        train it for 'short train' episodes, and evaluate it's performance on 5
        make sure agent and memory is discarded after evaluation
        '''
        fitness.append(np.random.uniform(), individual)
    return fitness

def make_population():
    '''
    Each machine generates it's own population. 
    Assume config.json is mirrored across nodes
    '''

    pop_size    = config['Population_Size']
    gene_param   = config['Genetic_Parameters']
    low_NHU, high_NHU       = gene_param['Num_Hidden_Units_Range']
    n_Layers                = gene_param['Max_Conv_Layers']
    kernel_max, stride_Max  = gene_param['Layer_KS_Limits']

    rand_h_size         = np.random.randint(low_NHU, high_NHU, pop_size)
    rand_layer_count    = np.random.randint(2, n_Layers+1, pop_size)
    rand_kern           = np.random.randint(2, kernel_max+1, pop_size * n_Layers)
    rand_stride         = np.random.randint(2, stride_Max+1, pop_size * n_Layers)
    rand_kern           = np.reshape(rand_kern, [pop_size, -1])
    rand_stride         = np.reshape(rand_stride, [pop_size, -1])
    return list(zip(rand_h_size, rand_layer_count, rand_kern, rand_stride))


if __name__== '__main__':
    config = json.load(open("config.json"))
    population = make_population()

    for i in range(config['Generations']):
        '''
            1) evaluate fitness of population
            2) while new_pop.size < "population_Size":
                individuals chosen with roulette wheel
                either:
                a) Mutate single individual and append to new pop
                b) crossover 2 individuals and append to new pop
            3) population = new_pop
        '''
        if i != 0 and i % config['GeneratMix_Intervalions'] == 0:
            mix_pop(population)



