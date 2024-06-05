import os
import numpy as np
import env_loader
import pandas as pd
from pettingzoo import ParallelEnv


def run_env(env: ParallelEnv, dict_agents, num_episodes, max_steps, seed):
  total_returns = {a_name: [] for a_name in dict_agents}
  for _ in range(num_episodes):
    jo_obs, infos = env.reset(seed)
    jo_p_lat = {aname: dict_agents[aname].PREV_LATENT for aname in env.agents}
    dummy_aux = {aname: dict_agents[aname].PREV_AUX for aname in env.agents}
    episode_rewards = {aname: 0 for aname in env.agents}
    done = False

    n_steps = 0
    while not done and n_steps < max_steps:
      jo_lat = {}
      jo_action = {}
      for a_name in env.agents:
        agent = dict_agents[a_name]
        latent, action = agent.choose_action(jo_obs[a_name],
                                             jo_p_lat[a_name],
                                             prev_aux=dummy_aux[a_name],
                                             sample=False)
        jo_lat[a_name] = latent
        jo_action[a_name] = action

      jo_obs, rewards, dones, truncates, infos = env.step(jo_action)
      n_steps += 1
      for a_name in env.agents:
        episode_rewards[a_name] += rewards[a_name]
        if dones[a_name] or truncates[a_name]:
          done = True

      jo_p_lat = jo_lat

    for a_name in env.agents:
      total_returns[a_name].append(episode_rewards[a_name])

  return total_returns


def compute_hri_reward(env_name, alg_name, learnt_agent_idx, supervision,
                       model_number, num_episodes):
  env, experts = env_loader.load_env(env_name)

  agent = env_loader.load_agent(env_name, alg_name, learnt_agent_idx,
                                supervision, model_number)

  agents = dict(experts)
  agents[learnt_agent_idx] = agent

  total_returns = run_env(env, agents, num_episodes, 1000, 0)

  agent_names = list(agents.keys())
  np_returns = np.array([total_returns[a_name] for a_name in agent_names])
  mean_returns = np.mean(np_returns, axis=1)
  std_returns = np.std(np_returns, axis=1)

  total_returns = np.sum(np_returns, axis=0)
  mean_total_return = np.mean(total_returns)
  std_total_return = np.std(total_returns)

  return mean_total_return, std_total_return, mean_returns, std_returns


if __name__ == "__main__":
  env_names = [
      "LaborDivision3", "LaborDivision2", "PO_Movers-v2", "PO_Flood-v2"
  ]
  alg_supervision = [("iiql", 0.0), ("mahil", 0.0), ("mahil", 0.2),
                     ("magail", 0.0), ("maogail", 0.0), ("maogail", 0.2)]
  learnt_agent_idx = [0, 1]
  model_number = [1, 2, 3]

  num_episodes = 30

  columns = [
      "env", "alg", "sv", "learnt_agent", "model_num", "mean_return_sum",
      "std_return_sum", "mean_return_0", "mean_return_1", "std_return_0",
      "std_return_1"
  ]

  list_results = []
  for env_name in env_names:
    for alg_name, supervision in alg_supervision:
      for agent_idx in learnt_agent_idx:
        for model_num in model_number:
          (mean_total_return, std_total_return, mean_returns,
           std_returns) = compute_hri_reward(env_name, alg_name, agent_idx,
                                             supervision, model_num,
                                             num_episodes)
          list_results.append(
              (env_name, alg_name, supervision, agent_idx, model_num,
               mean_total_return, std_total_return, mean_returns[0],
               mean_returns[1], std_returns[0], std_returns[1]))

  df = pd.DataFrame(list_results, columns=columns)

  cur_dir = os.path.dirname(__file__)
  save_name = os.path.join(cur_dir, "hri_reward.csv")
  df.to_csv(save_name, index=False)
