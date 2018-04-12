#run 1 thread per node on network
/usr/lib64/openmpi/bin/mpirun -pernode -hostfile ./machinefile python3 mpi.py
