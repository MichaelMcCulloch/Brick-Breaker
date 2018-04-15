import numpy as np

from Network import Q_Learner

class Agent():
    def __init__(self, traits, env):
        self.H_Size = traits['Hidden_Unit_Size']
        self.Layer_Count = traits['Layer_Count']
        self.Kernel_Size = traits['Kernel_Size'][0:self.Layer_Count]
        self.Stride_Length = traits['Stride_Length'][0:self.Layer_Count]

        
        self.mainQ      = Q_Learner(60, 408, self.H_Size, self.Layer_Count, self.Kernel_Size, self.Stride_Length, 1, 4, "MAIN")
        self.targetQ    = Q_Learner(60, 408, self.H_Size, self.Layer_Count, self.Kernel_Size, self.Stride_Length, 1, 4, "MAIN")

    def train(self, episode_count):
        pass

    def evaluate(self, num_tests=5):
        return np.random.uniform()