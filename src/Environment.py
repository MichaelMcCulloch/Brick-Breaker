import sys
import numpy as np
import vizdoom as vd

class Environment():
    def __init__(self, config):
        self.screen_shape = [60, 108, 3]
        #play random episodes to agent
        self.game = self._create_game(config.Environment_Type)

        self.n_feat = 1
        self.n_act = 4
        self.frame_skip = config.Frame_Skip_Count

    def _create_game(self, game_type):
    
        if game_type == 'Random':
            pass
        else:
            game = vd.DoomGame()
            game.load_config(game_type + '.cfg')

            walls = self._parse_walls(game_type + '.wad')
            game.clear_available_game_variables
            game.add_available_game_variable(vd.GameVariable.POSITION_X)  # 0
            game.add_available_game_variable(vd.GameVariable.POSITION_Y)  # 1
            game.add_available_game_variable(vd.GameVariable.POSITION_Z)  # 2

            game.add_available_game_variable(vd.GameVariable.KILLCOUNT)   # 3
            game.add_available_game_variable(vd.GameVariable.DEATHCOUNT)  # 4
            game.add_available_game_variable(vd.GameVariable.ITEMCOUNT)   # 5
            
            game.set_labels_buffer_enabled(True)    

            game.init()
            return game, walls
    def _parse_walls(self, wad_file)
    '''
    Returns a state and features
    '''
    def reset(self):
        pass:

    '''
    Returns a state, features, reward and done
    '''
    def step(self, action):
        pass:
    
    def close(self):
        pass