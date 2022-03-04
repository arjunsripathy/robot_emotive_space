import numpy as np
import torch
import sys
sys.path.append('../')

from full_cassie_env import FullCassieEnv
from style_net import StyleNetConfig
from interface import Interface

# Cassie Interface

class FullCassieInterface(Interface):
    def __init__(self, config):
        super().__init__(FullCassieEnv(config), config)

if (__name__ == '__main__'):
    config = StyleNetConfig(STATE_DIM = 13, LM = [])
    config.EXT_OPT = True
    config.CM_WAYPT_HID_DIMS = [16]
    config.CM_TRAJ_HID_DIMS = [32]
    interface = FullCassieInterface(config)
    interface.run()
