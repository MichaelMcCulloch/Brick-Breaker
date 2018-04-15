import json

class Config():
    def __init__(self, file):
        with open(file) as f:
            self.config = json.load(f)
    
        self.Genetic_Parameters = self.config["Genetic_Parameters"] 
        self.Hidden_Unit_Range  = self.Genetic_Parameters["Hidden_Unit_Range"]
        self.Hidden_Unit_Lower  = self.Hidden_Unit_Range[0]
        self.Hidden_Unit_Upper  = self.Hidden_Unit_Range[1]

        self.Conv_Layers_Max    = self.Genetic_Parameters['Conv_Layers_Max']
        self.Kernel_Size_Max    = self.Genetic_Parameters['Kernel_Size_Max']
        self.Stride_Length_Max  = self.Genetic_Parameters['Stride_Length_Max']
        self.Num_Filter_Range   = self.Genetic_Parameters['Num_Filter_Range']
        self.Filter_Lower       = self.Num_Filter_Range[0]
        self.Filter_Upper       = self.Num_Filter_Range[1]

        self.Population_Size    = self.config["Population_Size"]
        self.Generations        = self.config["Generations"]
        self.Mix_Interval       = self.config["Mix_Interval"]
        self.Mutation_Prob      = self.config["Mutation_Prob"]
        self.Memory_Max_Bytes   = self.config["Memory_Max_Bytes"]
        self.Short_Train        = self.config["Short_Train"]
        self.Long_Train         = self.config["Long_Train"]
        self.Recurrence         = self.config["Recurrence"]
        
        self.Batch_Size         = self.Recurrence["Batch_Size"]
        self.Sequence_Length    = self.Recurrence["Sequence_Length"]
        self.Ignore_Up_To       = self.Recurrence["Ignore_Up_To"]
        self.Episode_Length_Max = self.config["Episode_Length_Max"]

