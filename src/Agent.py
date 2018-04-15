import numpy as np

from Network import Q_Learner

class Agent():
    def __init__(self, traits, env, config):
        self.H_Size = traits[config.key_HUS]
        self.Layer_Count = traits[config.key_LC]
        self.Kernel_Size = traits[config.key_KS][0:self.Layer_Count]
        self.Stride_Length = traits[config.key_SL][0:self.Layer_Count]
        self.Num_Filter = traits[config.key_NF][0:self.Layer_Count]

        
        self.mainQ      = Q_Learner(60, 408, self.H_Size, self.Layer_Count, self.Kernel_Size, self.Stride_Length,self.Num_Filter, 1, 4, "MAIN")
        self.targetQ    = Q_Learner(60, 408, self.H_Size, self.Layer_Count, self.Kernel_Size, self.Stride_Length,self.Num_Filter, 1, 4, "MAIN")

    def train(self, episode_count):
        pass

    def evaluate(self, num_tests=5):
        return np.random.uniform()