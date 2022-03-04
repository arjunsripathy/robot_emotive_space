import numpy as np
import torch
import pyglet
from pyglet.window import key
import sys
import threading
import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append('../')

from cart_env import CartEnv
from interface import Interface
from trajectory import Trajectory
from style_net import StyleNetConfig

# VacuumBot Interface

class CartInterface(Interface):
    def __init__(self, config):
        super().__init__(CartEnv(config), config)

if (__name__ == '__main__'):
    config = StyleNetConfig(STATE_DIM = 7, LM = [1e4, 1e4])
    interface = CartInterface(config)
    interface.run()
