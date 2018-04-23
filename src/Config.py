import json


class Config():
    def __init__(self, file):
        with open(file) as f:
            self.config = json.load(f)

            self.Perform_GA = False
            self.Search_Hidden_Units = False
            self.Search_Conv_Layers = False
            self.Search_Kernel_Size = False
            self.Search_Stride_Length = False
            self.Search_Num_Filters = False

            if self.config['Genetic_Parameters'] != None:
                self.Genetic_Parameters = self.config['Genetic_Parameters']
                if 'Hidden_Unit_Range' in self.Genetic_Parameters and self.Genetic_Parameters['Hidden_Unit_Range'] != None:
                    self.Search_Hidden_Units = True
                    self.key_HUS = 'Hidden_Unit_Size'
                    self.Hidden_Unit_Range = self.Genetic_Parameters['Hidden_Unit_Range']

                if 'Conv_Layers_Max' in self.Genetic_Parameters and self.Genetic_Parameters['Conv_Layers_Max'] != None:
                    self.key_LC = 'Layer_Count'
                    self.Conv_Layers_Min = 2
                    self.Conv_Layers_Max = self.Genetic_Parameters['Conv_Layers_Max']
                else:
                    self.Conv_Layers_Min = 2
                    self.Conv_Layers_Max = 2

                if self.Conv_Layers_Max != self.Conv_Layers_Min:
                    self.Search_Conv_Layers = True

                if 'Kernel_Size_Max' in self.Genetic_Parameters and self.Genetic_Parameters['Kernel_Size_Max'] != None:
                    self.Search_Kernel_Size = True
                    self.key_KS = 'Kernel_Size'
                    self.Kernel_Size_Max = self.Genetic_Parameters['Kernel_Size_Max']

                if 'Stride_Length_Max' in self.Genetic_Parameters and self.Genetic_Parameters['Stride_Length_Max'] != None:
                    self.Search_Stride_Length = True
                    self.key_SL = 'Stride_Length'
                    self.Stride_Length_Max = self.Genetic_Parameters['Stride_Length_Max']

                if 'Num_Filter_Range' in self.Genetic_Parameters and self.Genetic_Parameters['Num_Filter_Range']  != None:
                   self.Search_Num_Filters                                                                         = True
                   self.key_NF                                                                                     = 'Num_Filter'
                   self.Num_Filter_Range                                                                           = self.Genetic_Parameters['Num_Filter_Range']

                if self.Search_Hidden_Units or self.Search_Conv_Layers or self.Search_Kernel_Size or self.Search_Stride_Length or self.Search_Num_Filters:
                    self.Perform_GA = True

            self.Population_Size  = self.config['Population_Size']
            self.Generations      = self.config['Generations']
            self.Mix_Interval     = self.config['Mix_Interval']
            self.Mutation_Prob    = self.config['Mutation_Prob']
            self.Memory_Max_Bytes = self.config['Memory_Max_Bytes']
            self.Episode_Size     = self.config['Episode_Size']
            self.Short_Train      = self.config['Short_Train']
            self.Long_Train       = self.config['Long_Train']

            self.Annealing_Steps = self.config['Annealing_Steps']
            self.Noise           = self.config['Noise']

            self.Recurrence = self.config['Recurrence']

            self.Batch_Size       = self.Recurrence['Batch_Size']
            self.Sequence_Length  = self.Recurrence['Sequence_Length']
            self.Ignore_Up_To     = self.Recurrence['Ignore_Up_To']
            self.Update_Frequency = self.Recurrence['Update_Frequency']
            self.Update_Speed_Tau = self.Recurrence['Update_Speed_Tau']

            self.Episode_Length_Max = self.config['Episode_Length_Max']
            self.Frame_Skip_Count   = self.config['Frame_Skip_Count']
            self.Pretrain_Episodes  = self.config['Pretrain_Episodes']
            self.Summary_Interval   = self.config['Summary_Interval']
            self.Memory_Capacity    = self.config['Memory_Capacity']
