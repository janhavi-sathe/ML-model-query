import os
import pickle
from collections import defaultdict
import gymnasium
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
import numpy as np
import cv2
from PIL import Image
from pettingzoo import ParallelEnv


class LaborDivision(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "multi_professionals"}

  def __init__(self, targets=[], n_agents=2, render_mode=None):
    """
        The init method takes in environment arguments and should define 
        the following attributes:
        - possible_agents
        - render_mode

        obs: (
          position, {teammate_rel_position}, progressing, {target_position} 
          )
    """

    self.render_mode = render_mode
    self.possible_agents = [idx for idx in range(n_agents)]

    VISIBLE_RADIUS = 2
    WORLD_HALF_SIZE_X = 5
    WORLD_HALF_SIZE_Y = 5

    self.vis_rad = VISIBLE_RADIUS
    self.half_sz_x = WORLD_HALF_SIZE_X
    self.half_sz_y = WORLD_HALF_SIZE_Y

    self.list_target = targets

    for agent in self.possible_agents:
      self.action_spaces[agent] = Box(low=-1,
                                      high=1,
                                      shape=(2, ),
                                      dtype=np.float32)
      low = ([-self.half_sz_x, -self.half_sz_y] +
             [-self.vis_rad, -self.vis_rad] * n_agents + [0] +
             list(np.array(self.list_target).reshape(-1)))

      self.observation_spaces[agent] = Box(low=np.array(low))
