#!/usr/bin/env python
import numpy as np

"""
Parallel Hello World
"""

from mpi4py import MPI
import sys

COMM = MPI.COMM_WORLD

size = COMM.Get_size()
rank = COMM.Get_rank()
name = MPI.Get_processor_name()


#sys.stdout.write( "Hello, World! I am process %d of %d on %s.\n" % (rank, size, name))

ranks = MPI.COMM_WORLD.allgather(rank)
scat = MPI.COMM_WORLD.scatter(np.reshape(ranks, [size,-1]))
print(ranks)
COMM.barrier()
print(scat)

