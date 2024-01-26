from .mahil import MAHIL
from omegaconf import DictConfig
from pettingzoo.utils.env import ParallelEnv
from gym.spaces import Discrete, Box


def make_mahil_agent(config: DictConfig, env: ParallelEnv, agent_idx):

  agent_name = env.agents[agent_idx]
  latent_dim = config.dim_c[agent_idx]
  if isinstance(env.observation_spaces[agent_name], Discrete):
    obs_dim = env.observation_spaces[agent_name].n
    discrete_obs = True
  else:
    obs_dim = env.observation_spaces[agent_name].shape[0]
    discrete_obs = False

  list_aux_dim = []
  list_discrete_aux = []
  for name in env.agents:
    if not (isinstance(env.action_spaces[name], Discrete)
            or isinstance(env.action_spaces[name], Box)):
      raise RuntimeError(
          "Invalid action space: Only Discrete and Box action spaces supported")

    if isinstance(env.action_spaces[name], Discrete):
      tmp_action_dim = env.action_spaces[name].n
      tmp_discrete_act = True
    else:
      tmp_action_dim = env.action_spaces[name].shape[0]
      tmp_discrete_act = False

    if name == agent_name:
      action_dim = tmp_action_dim
      discrete_act = tmp_discrete_act

    if config.use_auxiliary_obs is True:
      list_aux_dim.append(tmp_action_dim)
      list_discrete_aux.append(tmp_discrete_act)

  agent = MAHIL(config, obs_dim, action_dim, latent_dim, tuple(list_aux_dim),
                discrete_obs, discrete_act, tuple(list_discrete_aux))
  return agent
