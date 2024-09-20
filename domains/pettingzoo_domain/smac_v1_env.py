import os
import numpy as np
import functools
from gymnasium.spaces import Box, Discrete
from pettingzoo_domain.haven_envs.starcraft2.starcraft2 import StarCraft2Env
from pettingzoo.utils.env import ParallelEnv


class SMAC_V1(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "smac_v1"}

  def __init__(self, map_name, seed, max_length) -> None:
    super().__init__()

    self.env = StarCraft2Env(map_name=map_name, seed=seed)

    self.max_length = max_length
    self.possible_agents = list(range(self.env.n_agents))
    self.agents = self.possible_agents[:]

    observation_size = self.env.get_obs_size()
    self.observation_spaces = {}
    for name in self.agents:
      self.observation_spaces[name] = Box(low=-1,
                                          high=1,
                                          shape=(observation_size, ),
                                          dtype=np.float32)

    n_actions = self.env.get_total_actions()
    self.action_spaces = {}
    for name in self.agents:
      self.action_spaces[name] = Discrete(n_actions)

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.observation_spaces[agent]

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]

  def close(self):
    self.env.close()

  def reset(self, seed=None, options=None):
    # seed cannot be set here in SMACv1
    obs, state = self.env.reset()

    self.agents = self.possible_agents[:]
    self.cur_step = 0

    infos = []
    avail_actions = self.env.get_avail_actions()
    for idx in range(self.env.n_agents):
      infos.append({"avail_actions": np.array(avail_actions[idx])})

    return obs, infos

  def step(self, actions):
    list_actions = []
    for aname in self.agents:
      list_actions.append(actions[aname])

    reward, terminated, info = self.env.step(list_actions)

    obs = self.env.get_obs()
    avail_actions = self.env.get_avail_actions()

    new_info = {}
    for aname in self.agents:
      new_info[aname] = {}
      # copy same info to all agents
      for key, val in info.items():
        new_info[aname][key] = val[aname]

      new_info[aname]["avail_actions"] = np.array(avail_actions[aname])

    self.cur_step += 1

    trunc = self.cur_step >= self.max_length
    if terminated:
      dones = {agent: True for agent in self.agents}
      truncs = {agent: False for agent in self.agents}
    else:
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}

    rewards = {agent: reward for agent in self.agents}

    return obs, rewards, dones, truncs, new_info

  def save_replay(self):
    self.env.save_replay()


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
