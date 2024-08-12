import numpy as np
from omegaconf import OmegaConf
import torch
import os
import pandas as pd
from aic_core.utils.result_utils import hamming_distance
from aic_ml.MAHIL.helper.utils import load_trajectories
import run_btil


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


def load_btil_agent(model_dir, env_name, agent_idx, model_num):
  prefix = env_name + "_btil_svi_"
  postfix = f"_a{agent_idx + 1}_s{model_num}"

  policy_name = prefix + "policy" + postfix
  tx_name = prefix + "tx" + postfix
  bx_name = prefix + "bx" + postfix

  policy = np.load(os.path.join(model_dir, policy_name + ".npy"))
  tx = np.load(os.path.join(model_dir, tx_name + ".npy"))
  bx = np.load(os.path.join(model_dir, bx_name + ".npy"))

  # policy: X x O x A
  # tx: X x O x X
  # bx: O x X
  return policy, tx, bx


def infer_latent_result_btil(model_dir, env_name, alg_name, agent_idx,
                             supervision, model_num, trajectories):

  policy, tx, bx = load_btil_agent(model_dir, env_name, agent_idx, model_num)

  agent_trajs = trajectories[agent_idx]

  list_inferred_x = []
  for i_e in range(len(agent_trajs)):
    states, actions, _ = list(zip(*agent_trajs[i_e]))

    inferred_x, _ = agent.infer_mental_states(states, actions, [])
    list_inferred_x.append(inferred_x)

  ham_dists, lengths = get_stats_about_x(list_inferred_x,
                                         agent_trajs["latents"])
  accuracy = 1 - np.sum(ham_dists) / np.sum(lengths)

  return accuracy


def load_test_data(env_name):
  cur_dir = os.path.dirname(__file__)
  data_dir = os.path.join(cur_dir, "test_data")
  if env_name == "PO_Flood-v2":
    data_path = os.path.join(data_dir, "PO_Flood-v2_100.pkl")
  elif env_name == "PO_Movers-v2":
    data_path = os.path.join(data_dir, "PO_Movers-v2_100.pkl")
  else:
    raise ValueError(f"Unknown env_name: {env_name}")

  trajectories = load_trajectories(data_path, 100)

  return trajectories


if __name__ == "__main__":
  env_names = ["PO_Movers-v2", "PO_Flood-v2"]
  alg_supervisions = [("btil", 0.2)]
  agent_idxs = [0, 1]
  model_numbers = [1]

  columns = ["env", "alg", "sv", "agent_idx", "model_num", "accuracy"]

  list_results = []
  for env_name in env_names:
    trajectories = load_test_data(env_name)
    if env_name == "PO_Movers-v2":
      converter = run_btil.Converter_PO_Movers()
    elif env_name == "PO_Flood-v2":
      converter = run_btil.Converter_PO_Flood()
    else:
      raise NotImplementedError()

    btil_trajs = converter.convert_trajectories(trajectories, 0)

    for alg_name, supervision in alg_supervisions:
      for agent_idx in agent_idxs:
        for model_num in model_numbers:
          accuracy = infer_latent_result_btil(model_dir, env_name, alg_name,
                                              agent_idx, supervision, model_num,
                                              btil_trajs)
          list_results.append(
              (env_name, alg_name, supervision, agent_idx, model_num, accuracy))

  df = pd.DataFrame(list_results, columns=columns)

  cur_dir = os.path.dirname(__file__)
  save_name = os.path.join(cur_dir, "infer_latent_result.csv")
  df.to_csv(save_name, index=False)
