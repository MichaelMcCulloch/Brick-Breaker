#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import numpy as np
import sys

from Config import Config
from Agent import Agent
from Memory import Replay_Buffer
from common import processState, chunks
import gym

COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()
name = MPI.Get_processor_name()

config = None



'''
Share population among workers
'''
def mix_pop(population):
    # gather() population into root node
    
    all_pop = COMM.gather(population, root=0)
    
    print("MIXING")
    if rank == 0:
        # flatten gathered list of lists
        all_pop = [b for val in all_pop for b in val]
        print(all_pop[0])
        np.random.shuffle(all_pop)                 # shuffle population
        # chunk it for transmission
        all_pop = list(chunks(all_pop, len(all_pop)//size))
    new_pop = COMM.scatter(all_pop, root=0)
    return new_pop

def find_best_member(population):
    all_pop = COMM.gather(population, root=0)
    if rank == 0:
        all_pop = [b for val in all_pop for b in val]
        all_pop.sort(key=lambda p: p[0], reverse = True)
        best = all_pop[0]
    else:
        best = None
    best = COMM.bcast(best, root=0)
    return best[1]

'''
Generate the initial population. 
Random values are chosen for number of hidden units, Layer Count, Kernel Sizes, Kernel Stride Length, and number of output filters
'''
def make_population():
    pop_size = config.Population_Size
    toZip = list()
    initial_fitness = np.repeat(None, pop_size)
    n_layers = None
    if config.Search_Hidden_Units:
        low_NHU, high_NHU = config.Hidden_Unit_Range
        rand_h_size = np.random.randint(low_NHU, high_NHU // 2, pop_size)
        toZip.append((config.key_HUS, rand_h_size * 2))

    if config.Search_Conv_Layers:
        n_layers = config.Conv_Layers_Max
        rand_layer_count = np.random.randint(2, config.Conv_Layers_Max + 1, pop_size)
        toZip.append((config.key_LC, rand_layer_count))
    else:
        n_layers = 2

    if config.Search_Kernel_Size:
        rand_kern = np.random.randint(2, config.Kernel_Size_Max+1, pop_size * n_layers)
        rand_kern = np.reshape(rand_kern, [pop_size, -1])
        toZip.append((config.key_KS, rand_kern))

    if config.Search_Stride_Length:
        rand_stride = np.random.randint(2, config.Stride_Length_Max+1, pop_size * n_layers)
        rand_stride = np.reshape(rand_stride, [pop_size, -1])
        toZip.append((config.key_SL, rand_stride))
    if config.Search_Num_Filters:
        filter_range = config.Num_Filter_Range
        rand_filter = np.random.randint(filter_range[0], filter_range[1] + 1, pop_size * n_layers)
        rand_filter = np.reshape(rand_filter, [pop_size, -1])
        toZip.append((config.key_NF, rand_filter))

    individuals = list()
    for p in range(0,pop_size):
        individual = dict()
        for k in toZip:
            individual.update({k[0]: k[1][p]})
        individuals.append(individual)
    return list(zip(initial_fitness,individuals))

'''
Choose a random member from a population, 
with greater probability of choosing more fit members
'''
def roulette(population):
    max = sum([p[0] for p in population])
    pick = np.random.uniform(0, max)
    current = 0
    for p in population:
        current += p[0]
        if current > pick:
            return p


def mutate(individual):
    new_member = individual.copy()
    key = np.random.choice(list(new_member))
    if key == config.key_HUS:
        min = config.Hidden_Unit_Range[0]
        max = config.Hidden_Unit_Range[1] // 2
    if key == config.key_LC:
        min = config.Conv_Layers_Min
        max = config.Conv_Layers_Max + 1
    if key == config.key_KS:
        min = 2
        max = config.Kernel_Size_Max 
    if key == config.key_SL:
        min = 2
        max = config.Stride_Length_Max
    if key == config.key_NF:
        min = config.Num_Filter_Range[0]
        max = config.Num_Filter_Range[1]
        
    #special handling for arrays
    if key == config.key_KS or key == config.key_SL or key == config.key_NF:
        rand_Index = np.random.randint(0, new_member[config.key_LC] if config.Search_Conv_Layers else 2)
        arr = new_member[key]
        oldval = arr[rand_Index]
        newval = oldval
        while newval == oldval:
            newval = np.random.randint(min, max)
        arr[rand_Index] = newval
        new_member.update({key : arr})
    else:
        oldval = new_member[key]
        newval = oldval
        while newval == oldval:
            newval = np.random.randint(min, max)
        if key == config.key_HUS: newval *= 2
        new_member.update({key : newval})
    
    return new_member

def cross(A, B):

    k = list(A.keys())
    r = np.repeat(0, len(k))
    while not np.any(r) and not np.all(r):
        r = np.random.randint(0, 2, len(k))
    new_A = A.copy()
    new_B = B.copy()
    
    keys = list()
    for idx, bool in enumerate(r):
        if bool:
            keys.append(k[idx])
            
    for key in keys:
        temp = new_B[key]
        new_B.update({key: new_A[key]})
        new_A.update({key: temp})

    return new_A, new_B


def breed(gene_pool, target_size, p_mutate=0.8):

    new_pop = []
    gene_pool.sort(key=lambda p: p[0], reverse = True)
    new_pop.extend(gene_pool[0:target_size//4])  # elitism
    while len(new_pop) < target_size:
        r = np.random.uniform()
        if r < p_mutate:
            succ = roulette(gene_pool)
            mut = mutate(succ[1])
            new_pop.append((None, mut))
        else:
            succ_A = roulette(gene_pool)
            succ_B = roulette(gene_pool)
            while succ_A == succ_B:
                succ_B = roulette(gene_pool)
            new_A, new_B = cross(succ_A[1], succ_B[1])
            b = np.random.randint(0, 1)
            new_pop.append((None, new_A)) if b == 0 else new_pop.append(
                (None, new_B))
    return new_pop

if __name__ == '__main__':
    config = Config("no_genetic.json")
    env = gym.make('Breakout-v0')
    best_traits = dict()
    if config.Perform_GA:
        population = make_population()

        for gen in range(config.Generations):

            # Occasionally mix population among workers
            if gen != 0 and gen % config.Mix_Interval == 0:
                population = mix_pop(population)

            # generate a list of candidates
            candidates = []
            for individual in population:
                fitness = individual[0]
                traits = individual[1]

                # don't evaluate an individual more than once
                if fitness != None:
                    candidates.append(individual)
                    continue

                # creating an agent may fail if parameters are incompatible with network!
                agent = Agent(env, config, **traits)
                
                agent.train(episode_count=config.Short_Train)
                fitness = agent.evaluate()
                agent.delete()
                candidates.append((fitness, traits))

            population = breed(candidates, len(candidates),
                            p_mutate=config.Mutation_Prob)
        best_traits = find_best_member(candidates)
        print("Best parameters Found =", best[1], "with fitness", best[0])
    
    if rank == 0:
        traits = best_traits
        agent = Agent(env, config, **traits ,save=True)
        agent.train(episode_count=config.Long_Train)