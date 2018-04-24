from Agent import Agent
from Config import Config
import gym
import ast

import sys

if len(sys.argv) < 2:
    print("Please provide a model name")
else:
    if len(sys.argv) == 3:
        episodes = int(sys.argv[2])
    else: 
        episodes = 1
    model_name = sys.argv[1]
    name_ext = model_name.split('/')[-1]
    name = name_ext.split('.')[0].replace('(', '[').replace(')',']')
    params = name.split('_')
    HSU = int(params[0])
    LC = int(params[1])
    KS = ast.literal_eval(params[2])
    SL = ast.literal_eval(params[3])
    NF = ast.literal_eval(params[4])

    env = gym.make('Breakout-v0')
    config = Config('config.json')
    agent = Agent(env, config, Hidden_Unit_Size=HSU, Layer_Count=LC, Kernel_Size=KS, Stride_Length=SL, Num_Filter=NF, path=model_name)

    for i in range(episodes):
        agent.play_episode(random=False, render=True)
