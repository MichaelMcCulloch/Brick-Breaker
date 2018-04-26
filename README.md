# DistributedDoomSlayer
## Python 3.6 Dependencies
1. numpy
2. cv2
3. OpenAI Gym
4. Tensorflow 1.6
5. mpi4py and MPI 3.0 implementation

## Usage
As training progresses, the agent saves copies of itself named with it's hyperparameters

### Training
% mpirun -pernode -hostfile ./machinefile python3 ./main_train.py

### Evaluation
% python3 ./main_play.py path/to/model numberOfGames
