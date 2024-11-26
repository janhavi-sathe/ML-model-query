import numpy as np
from omegaconf import OmegaConf
import torch
import os
import pandas as pd
import env_loader
from aicoach.algs.utils.result_utils import hamming_distance
from dtil.helper.utils import load_trajectories


def get_stats_about_x(list_inferred_x, list_true_x):
  dis_array = []
  length_array = []
  for i_e, inferred_x in enumerate(list_inferred_x):
    res = hamming_distance(inferred_x, list_true_x[i_e])
    dis_array.append(res)
    length_array.append(len(inferred_x))

  dis_array = np.array(dis_array)
  length_array = np.array(length_array)
  return dis_array, length_array


def infer_latent_result(env_name, alg_name, agent_idx, supervision, model_num,
                        trajectories):

  agent = env_loader.load_agent(env_name, alg_name, agent_idx, supervision,
                                model_num)

  agent_trajs = trajectories[agent_idx]

  list_inferred_x = []
  for i_e in range(len(agent_trajs["states"])):
    states = agent_trajs["states"][i_e]
    actions = agent_trajs["actions"][i_e]
    inferred_x, _ = agent.infer_mental_states(states, actions, [])
    list_inferred_x.append(inferred_x)

  ham_dists, lengths = get_stats_about_x(list_inferred_x,
                                         agent_trajs["latents"])
  accuracy = 1 - np.sum(ham_dists) / np.sum(lengths)

  return accuracy


def load_test_data(env_name):
  cur_dir = os.path.dirname(__file__)
  data_dir = os.path.join(cur_dir, "test_data")
  if env_name == "LaborDivision2-v2":
    data_path = os.path.join(data_dir, "LaborDivision2-v2_100.pkl")
  elif env_name == "LaborDivision3-v2":
    data_path = os.path.join(data_dir, "LaborDivision3-v2_100.pkl")
  elif env_name == "PO_Flood-v2":
    data_path = os.path.join(data_dir, "PO_Flood-v2_100.pkl")
  elif env_name == "PO_Movers-v2":
    data_path = os.path.join(data_dir, "PO_Movers-v2_100.pkl")
  else:
    raise ValueError(f"Unknown env_name: {env_name}")

  trajectories = load_trajectories(data_path, 100)

  return trajectories


if __name__ == "__main__":
  env_names = [
      "LaborDivision3-v2", "LaborDivision2-v2", "PO_Movers-v2", "PO_Flood-v2"
  ]
  alg_supervisions = [("mahil", 0.2), ("maogail", 0.2)]
  agent_idxs = [0, 1]
  model_numbers = [1, 2, 3]

  columns = ["env", "alg", "sv", "agent_idx", "model_num", "accuracy"]

  list_results = []
  for env_name in env_names:
    trajectories = load_test_data(env_name)
    for alg_name, supervision in alg_supervisions:
      for agent_idx in agent_idxs:
        for model_num in model_numbers:
          accuracy = infer_latent_result(env_name, alg_name, agent_idx,
                                         supervision, model_num, trajectories)
          list_results.append(
              (env_name, alg_name, supervision, agent_idx, model_num, accuracy))

  df = pd.DataFrame(list_results, columns=columns)

  cur_dir = os.path.dirname(__file__)
  save_name = os.path.join(cur_dir, "infer_latent_result.csv")
  df.to_csv(save_name, index=False)
