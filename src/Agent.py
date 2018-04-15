import numpy as np

from Network import Q_Learner


class Agent():
    def __init__(self, env, config, Hidden_Unit_Size=300, Layer_Count=2, Kernel_Size=[8, 4], Stride_Length=[4, 2], Num_Filter=[32, 64]):
        
        
        
        
        self.mainQ      = Q_Learner(60, 408, Hidden_Unit_Size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, 1, 4, "MAIN")
        self.targetQ    = Q_Learner(60, 408, Hidden_Unit_Size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, 1, 4, "TARGET")

    def train(self, episode_count):
        pass

    def evaluate(self, num_tests=5):
        return np.random.uniform()
