import os
import pickle
import functools
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

  def __init__(self, targets, n_agents, render_mode=None):
    """
        The init method takes in environment arguments and should define 
        the following attributes:
        - possible_agents
        - render_mode

        obs: (position, {teammate_rel_position}, progressing)
    """

    self.render_mode = render_mode
    self.possible_agents = [idx for idx in range(n_agents)]

    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))

    VISIBLE_RADIUS = 2
    WORLD_HALF_SIZE_X = 5
    WORLD_HALF_SIZE_Y = 5

    self.vis_rad = VISIBLE_RADIUS
    self.half_sz_x = WORLD_HALF_SIZE_X
    self.half_sz_y = WORLD_HALF_SIZE_Y

    # define task world
    self.max_step = 100
    self.np_targets = np.array(targets)
    self.dict_agent_pos = {}
    #  -  target status: 0 means ready. positive values mean progress
    #                    negative values mean remaining time to get ready
    #                    each column corresponds to each agent.
    self.restock_time = 30
    self.max_progress = 5
    self.target_status = np.zeros((len(self.np_targets), n_agents))
    self.tolerance = 0.5

    # define observation and action spaces
    self.NOT_OBSERVED = np.array([self.vis_rad, self.vis_rad])
    obs_low = np.array([-self.half_sz_x, -self.half_sz_y] +
                       [-self.vis_rad, -self.vis_rad] * (n_agents - 1) + [0])
    obs_high = np.array([self.half_sz_x, self.half_sz_y] +
                        [self.vis_rad, self.vis_rad] * (n_agents - 1) + [1])
    for agent in self.possible_agents:
      self.action_spaces[agent] = Box(low=-1,
                                      high=1,
                                      shape=(2, ),
                                      dtype=np.float32)

      self.observation_spaces[agent] = Box(low=obs_low,
                                           high=obs_high,
                                           dtype=np.float32)

    # define visualization parameters
    self.canvas_sz = 300

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    # gymnasium spaces are defined and documented here:
    #    https://gymnasium.farama.org/api/spaces/
    return self.observation_spaces[agent]

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]

  def render(self):
    if self.render_mode is None:
      gymnasium.logger.warn(
          "You are calling render method without specifying any render mode.")
      return
    pass

  def close(self):
    pass

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

  def _closest_target(self, agent):
    tidx = -1
    min_dist = 99999
    pos = self.dict_agent_pos[agent]
    for idx, target in enumerate(self.np_targets):
      dist = np.linalg.norm(target - pos)
      if min_dist > dist:
        min_dist = dist
        tidx = idx

    return tidx

  def _compute_obs(self):
    # obs: (position, {teammate_rel_position}, progressing)

    dict_obs = {}
    for agent_me in self.possible_agents:
      my_pos = self.dict_agent_pos[agent_me]
      obs = [my_pos]
      # relative pos of other agents
      closest_agent = None
      min_dist = 99999
      for agent_ot in self.possible_agents:
        if agent_me == agent_ot:
          continue
        rel_pos = self.dict_agent_pos[agent_ot] - my_pos
        # if out of visible range, set it as not-observed
        dist = np.linalg.norm(rel_pos)
        if dist > self.vis_rad:
          rel_pos = self.NOT_OBSERVED
        if dist < min_dist:
          min_dist = dist
          closest_agent = agent_ot

        obs.append(rel_pos)

      # progressing?
      tidx = self._closest_target(agent_me)
      tar_pos = self.np_targets[tidx]
      dist2tar = np.linalg.norm(tar_pos - my_pos)
      distoth2tar = np.linalg.norm(tar_pos - self.dict_agent_pos[closest_agent])
      tar_status = self.target_status[tidx, self.agent_name_mapping[agent_me]]

      # to progress, an agent has to be close to the target,
      # there is no other agent around the target,
      # and the target has a job to progress
      progressing = 0
      if (dist2tar <= self.tolerance and distoth2tar > self.tolerance
          and tar_status >= 0 and tar_status < self.max_progress):
        progressing = 1
      obs.append(progressing)

      dict_obs[agent_me] = np.hstack(obs)

    return dict_obs

  def reset(self, seed=None, options=None):
    """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
    if seed is not None:
      self._seed(seed)

    self.agents = self.possible_agents[:]
    n_agents = len(self.agents)
    self.cur_step = 0

    margin = 0.5
    self.dict_agent_pos = {
        aname:
        (np.random.uniform(-self.half_sz_x + margin, self.half_sz_x - margin),
         np.random.uniform(-self.half_sz_y + margin, self.half_sz_y - margin))
        for aname in range(n_agents)
    }
    self.target_status = np.zeros((len(self.np_targets), n_agents))

    observations = self._compute_obs()
    infos = {agent: {} for agent in self.agents}

    return observations, infos

  def step(self, actions):
    """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

    self.cur_step += 1
    trunc = self.cur_step >= self.max_step

    next_agent_pos = {}
    for agent in self.possible_agents:
      pos = self.dict_agent_pos[agent] + np.array(actions[agent])
      pos[0] = min(self.observation_spaces[agent].high[0],
                   max(self.observation_spaces[agent].low[0], pos[0]))
      pos[1] = min(self.observation_spaces[agent].high[1],
                   max(self.observation_spaces[agent].low[1], pos[1]))
      next_agent_pos[agent] = pos

    PENALTY = -0.1
    POINT = 1

    reward = PENALTY


class DyadLaborDivision(LaborDivision):

  def __init__(self, targets, render_mode=None):
    super().__init__(targets, n_agents=2, render_mode=render_mode)


class TwoTargetDyadLaborDivision(DyadLaborDivision):

  def __init__(self, render_mode=None):
    super().__init__([(-4, 0), (4, 0)], render_mode)
