import dm_control
import numpy as np
from dm_control import suite
from dm_control import viewer

env = suite.load('manipulator', 'insert_ball')
viewer.launch(env)