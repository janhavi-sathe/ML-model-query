import os
from omegaconf import OmegaConf
from pettingzoo_domain.sc2_env import Protoss5v5, Terran5v5

from aic_ml.baselines.ma_ogail.model.agent import make_agent
from aic_ml.MAHIL.agent import make_mahil_agent
import hri_reward


def load_env(env_name):
  if env_name == "Protoss5v5":
    env = Protoss5v5()
  elif env_name == "Terran5v5":
    env = Terran5v5()
  else:
    raise ValueError(f"{env_name} is not supported")

  return env


def load_agent(env_name, aname):
  cur_dir = os.path.dirname(__file__)
  hri_results_dir = os.path.join(cur_dir, "result_hri", env_name)

  alg_name = "mahil"
  sv = 0.0
  number = 1

  list_log_dir = []
  if env_name == "Protoss5v5":
    list_log_dir.append("mahil/Seed3Sv0.0/2024-05-22_09-44-49")
  elif env_name == "Terran5v5":
    list_log_dir.append("mahil/Seed3Sv0.0/2024-05-22_09-45-18")
  else:
    raise ValueError(f"{env_name} is not supported")

  log_dir = os.path.join(hri_results_dir, list_log_dir[number - 1])

  config_path = os.path.join(log_dir, "log/config.yaml")
  config = OmegaConf.load(config_path)

  n_traj = int(config.n_traj)
  n_label = 0 if alg_name in ["iiql", "magail"] else int(n_traj * sv)
  model_name = env_name + f"_n{n_traj}_l{n_label}_best_{aname}"
  model_path = os.path.join(log_dir, f"model/{model_name}")

  env = load_env(env_name)
  env.reset()

  if alg_name == "iiql" or alg_name == "mahil":
    agent = make_mahil_agent(config, env, aname)
  elif alg_name == "magail":
    agent = make_agent(config, env, aname, False)
  elif alg_name == "maogail":
    agent = make_agent(config, env, aname, True)
  else:
    raise ValueError(f"Algorithm {alg_name} is not supported")
  agent.load(model_path)

  return agent


if __name__ == "__main__":
  env_name = "Terran5v5"
  env = load_env(env_name)
  env.reset()
  dict_agents = {}
  for aname in env.agents:
    dict_agents[aname] = load_agent(env_name, aname)

  total_return = hri_reward.run_env(env, dict_agents, 10, 1000, 0)
  env.save_replay()
