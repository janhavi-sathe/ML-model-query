import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def fill_na(res):

  idx = -1
  val = None
  # find the closest value
  for i in range(len(res)):
    if not np.isnan(res[i]):
      idx = i
      val = res[i]
      break

  res[:idx] = res[idx]
  for i in range(idx, len(res)):
    if np.isnan(res[i]):
      res[i] = val
    else:
      val = res[i]

  return res


def plot(csv_dir, domain_name):

  if domain_name == "LaborDivision3-v2":
    prefix = "ld3_"
    title = "Multi-Jobs-3"
    divide_by = 1
    xlim = 5e5
    down_sampling = 5
    maogailSV2_names = [
        'maogail_Seed1Sv0.2', 'maogail_Seed2Sv0.2', 'maogail_Seed3Sv0.2'
    ]
    maogailSV0_names = [
        'maogail_Seed1Sv0.0', 'maogail_Seed2Sv0.0', 'maogail_Seed3Sv0.0'
    ]
    mahilSV2_names = [
        'mahil_Seed1Sv0.2', 'mahil_Seed2Sv0.2', 'mahil_Seed3Sv0.2'
    ]
    mahilSV0_names = [
        'mahil_Seed1Sv0.0', 'mahil_Seed2Sv0.0', 'mahil_Seed3Sv0.0'
    ]
    iiql_names = ['iiql_Seed1', 'iiql_Seed2', 'iiql_Seed3']
    magail_names = ['magail_Seed1', 'magail_Seed2', 'magail_Seed3']

  elif domain_name == "LaborDivision2-v2":
    prefix = "ld2_"
    title = "Multi-Jobs-2"
    divide_by = 1
    xlim = 5e5
    down_sampling = 5
    maogailSV2_names = [
        'maogail_Seed1Sv0.2', 'maogail_Seed2Sv0.2', 'maogail_Seed3Sv0.2'
    ]
    maogailSV0_names = [
        'maogail_Seed1Sv0.0', 'maogail_Seed2Sv0.0', 'maogail_Seed3Sv0.0'
    ]
    mahilSV2_names = [
        'mahil_Seed1Sv0.2', 'mahil_Seed2Sv0.2', 'mahil_Seed3Sv0.2'
    ]
    mahilSV0_names = [
        'mahil_Seed1Sv0.0', 'mahil_Seed2Sv0.0', 'mahil_Seed3Sv0.0'
    ]
    iiql_names = ['iiql_Seed1', 'iiql_Seed2', 'iiql_Seed3']
    magail_names = ['magail_Seed1', 'magail_Seed2', 'magail_Seed3']
  elif domain_name == "PO_Movers-v2":
    prefix = "mov_"
    title = "Movers"
    divide_by = 2
    xlim = 3e5
    down_sampling = 10
    maogailSV2_names = [
        'maogail_Seed1Sv0.2', 'maogail_Seed2Sv0.2', 'maogail_Seed3Sv0.2'
    ]
    maogailSV0_names = [
        'maogail_Seed1Sv0.0', 'maogail_Seed2Sv0.0', 'maogail_Seed3Sv0.0'
    ]
    mahilSV2_names = [
        'mahil_shortSeed1Sv0.2', 'mahil_shortSeed2Sv0.2',
        'mahil_shortSeed3Sv0.2'
    ]
    mahilSV0_names = [
        'mahil_shortSeed1Sv0.0', 'mahil_shortSeed2Sv0.0',
        'mahil_shortSeed3Sv0.0'
    ]
    iiql_names = ['iiql_shortSeed1', 'iiql_shortSeed2', 'iiql_shortSeed3']
    magail_names = ['magail_tr2Seed1', 'magail_tr2Seed2', 'magail_tr2Seed3']
  elif domain_name == "PO_Flood-v2":
    prefix = "fld_"
    title = "Flood"
    divide_by = 2
    xlim = 3e5
    down_sampling = 10
    maogailSV2_names = [
        'maogail_tr2Seed1Sv0.2', 'maogail_tr2Seed2Sv0.2',
        'maogail_tr2Seed3Sv0.2'
    ]
    maogailSV0_names = [
        'maogail_tr2Seed1Sv0.0', 'maogail_tr2Seed2Sv0.0',
        'maogail_tr2Seed3Sv0.0'
    ]
    mahilSV2_names = [
        'mahil_shortSeed1Sv0.2', 'mahil_shortSeed2Sv0.2',
        'mahil_shortSeed3Sv0.2'
    ]
    mahilSV0_names = [
        'mahil_shortSeed1Sv0.0', 'mahil_shortSeed2Sv0.0',
        'mahil_shortSeed3Sv0.0'
    ]
    iiql_names = ['iiql_shortSeed1', 'iiql_shortSeed2', 'iiql_shortSeed3']
    magail_names = ['magail_tr2Seed1', 'magail_tr2Seed2', 'magail_tr2Seed3']
  elif domain_name == "Protoss5v5":
    prefix = "prt_"
    title = "Protoss5v5"
    divide_by = 5
    xlim = 5e5
    down_sampling = 5
    maogailSV0_names = [
        'maogail_Seed1Sv0.0', 'maogail_Seed2Sv0.0', 'maogail_Seed3Sv0.0'
    ]
    mahilSV0_names = [
        'mahil_Seed1Sv0.0', 'mahil_Seed2Sv0.0', 'mahil_Seed3Sv0.0'
    ]
    iiql_names = ['mahil_iiqlSeed1', 'mahil_iiqlSeed2', 'mahil_iiqlSeed3']
    magail_names = ['magail_Seed1', 'magail_Seed2', 'magail_Seed3']
  elif domain_name == "Terran5v5":
    prefix = "trn_"
    title = "Terran5v5"
    divide_by = 5
    xlim = 5e5
    down_sampling = 5
    maogailSV0_names = [
        'maogail_Seed1Sv0.0', 'maogail_Seed2Sv0.0', 'maogail_Seed3Sv0.0'
    ]
    mahilSV0_names = [
        'mahil_Seed1Sv0.0', 'mahil_Seed2Sv0.0', 'mahil_Seed3Sv0.0'
    ]
    iiql_names = ['mahil_iiqlSeed1', 'mahil_iiqlSeed2', 'mahil_iiqlSeed3']
    magail_names = ['magail_Seed1', 'magail_Seed2', 'magail_Seed3']
  else:
    pass

  dict_df = {}
  dict_df['iiql'] = pd.read_csv(os.path.join(csv_dir, prefix + "iiql.csv"))
  dict_df["magail"] = pd.read_csv(os.path.join(csv_dir, prefix + "magail.csv"))
  dict_df["mahilSV0"] = pd.read_csv(
      os.path.join(csv_dir, prefix + "mahilSV0.csv"))
  dict_df["maogailSV0"] = pd.read_csv(
      os.path.join(csv_dir, prefix + "maogailSV0.csv"))

  dict_names = {}
  dict_names['iiql'] = iiql_names
  dict_names['magail'] = magail_names
  dict_names['mahilSV0'] = mahilSV0_names
  dict_names['maogailSV0'] = maogailSV0_names

  if domain_name not in ["Protoss5v5", "Terran5v5"]:
    dict_df["mahilSV2"] = pd.read_csv(
        os.path.join(csv_dir, prefix + "mahilSV2.csv"))
    dict_df["maogailSV2"] = pd.read_csv(
        os.path.join(csv_dir, prefix + "maogailSV2.csv"))
    dict_names['mahilSV2'] = mahilSV2_names
    dict_names['maogailSV2'] = maogailSV2_names

  dict_labels = {
      'iiql': "IIQL",
      'magail': "MAGAIL",
      'mahilSV0': "DTIL",
      'mahilSV2': "DTIL-s",
      'maogailSV0': "MOG",
      'maogailSV2': "MOG-s"
  }

  dict_colors = {
      'iiql': "g",
      'magail': "y",
      'mahilSV0': "b",
      'mahilSV2': "c",
      'maogailSV0': "r",
      'maogailSV2': "m"
  }

  epi_steps = {}
  means = {}
  stds = {}
  for key, df in dict_df.items():

    res_group = []
    for run_name in dict_names[key]:
      res = df[run_name + ' - train/episode_reward']
      res = fill_na(res.values)
      res_group.append(res)
      epi_step = df['global_step'].values

    res_group = np.array(res_group)
    mean_res = np.mean(res_group, axis=0)
    std_res = np.std(res_group, axis=0)

    means[key] = mean_res
    stds[key] = std_res
    epi_steps[key] = epi_step

  plt.figure(figsize=(10, 6))

  for key, df in dict_df.items():
    if key in ["maogailSV2", "maogailSV0", "magail"]:
      down_sampling_ = 1
    else:
      down_sampling_ = down_sampling

    mean_res = means[key][::down_sampling_] / divide_by
    std_res = stds[key][::down_sampling_] / divide_by
    epi_step = epi_steps[key][::down_sampling_]
    plt.plot(epi_step, mean_res, label=dict_labels[key], color=dict_colors[key])
    plt.fill_between(epi_step,
                     mean_res - std_res,
                     mean_res + std_res,
                     color=dict_colors[key],
                     alpha=0.2)

    # plt.plot(df["global_step"][::down_sampling],
    #          df["Grouped runs - train/episode_reward"][::down_sampling] /
    #          divide_by,
    #          label=dict_labels[key],
    #          color=dict_colors[key])
    # plt.fill_between(
    #     df["global_step"][::down_sampling],
    #     df["Grouped runs - train/episode_reward__MIN"][::down_sampling] /
    #     divide_by,
    #     df["Grouped runs - train/episode_reward__MAX"][::down_sampling] /
    #     divide_by,
    #     color=dict_colors[key],
    #     alpha=0.2)

  plt.xlabel('Exploration Steps', fontsize=16)
  plt.ylabel('Task Reward', fontsize=16)
  plt.xlim(0, xlim)
  plt.title(title, fontsize=16)
  plt.legend()
  cur_dir = os.path.dirname(__file__)
  plt.savefig(cur_dir + f"/{domain_name}.png")
  # plt.show()


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
  csv_dir = os.path.join(cur_dir, "csv_files2")

  # LaborDivision2-v2  LaborDivision3-v2  PO_Movers-v2  PO_Flood-v2
  # Protoss5v5  Terran5v5
  plot(csv_dir, "Terran5v5")
  pass
