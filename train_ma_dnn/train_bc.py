from typing import List
import os
import stable_baselines3 as sb3
import gym
import random
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.util import logger as sb3_logger
from pettingzoo.utils.env import ParallelEnv  # noqa: F401
import wandb
import omegaconf
from aic_ml.MAHIL.helper.utils import load_trajectories


def eval_policy(env: ParallelEnv, list_policy: List[ActorCriticPolicy],
                num_episodes):
  n_agents = len(list_policy)
  total_timesteps = []
  total_returns = {a_name: [] for a_name in range(n_agents)}
  wins = []

  while len(total_timesteps) < num_episodes:
    done = False
    is_win = False
    episode_rewards = {a_name: 0 for a_name in range(n_agents)}
    episode_step = 0

    joint_obs, infos = env.reset()
    while not done:
      joint_actions = {}
      for a_name in range(n_agents):
        agent = list_policy[a_name]
        if "avail_actions" in infos[a_name]:
          available_actions = np.array(infos[a_name]["avail_actions"])
        else:
          available_actions = None
        with torch.no_grad():
          if available_actions is None:
            action, _ = agent.predict(joint_obs[a_name], deterministic=True)
          else:
            agent.set_training_mode(False)
            obs_tensor, _ = agent.obs_to_tensor(joint_obs[a_name])
            preprocessed_obs = preprocess_obs(
                obs_tensor,
                agent.observation_space,
                normalize_images=agent.normalize_images)
            features = agent.pi_features_extractor(preprocessed_obs)
            latent_pi = agent.mlp_extractor.forward_actor(features)
            mean_actions = agent.action_net(latent_pi)
            mean_actions[available_actions.reshape(mean_actions.shape) ==
                         0] = -1e10
            dist = agent.action_dist.proba_distribution(
                action_logits=mean_actions)
            action = dist.get_actions(deterministic=True)
            action = action.cpu().numpy().reshape(
                (-1, *agent.action_space.shape))
          joint_actions[a_name] = action

      joint_obs, rewards, dones, truncates, infos = env.step(joint_actions)

      episode_step += 1
      for a_name in env.agents:
        episode_rewards[a_name] += rewards[a_name]

        if dones[a_name] or truncates[a_name]:
          done = True

        if "won" in infos[a_name] and infos[a_name]["won"]:
          is_win = True

    for a_name in env.agents:
      total_returns[a_name].append(episode_rewards[a_name])
    total_timesteps.append(episode_step)
    wins.append(int(is_win))

  return total_returns, total_timesteps, wins


def train_bc(config, demo_path, log_dir, output_dir, cb_env_factory,
             log_interval, eval_interval, env_kwargs):
  env_name = config.env_name
  seed = config.seed
  num_episodes = config.num_eval_episodes
  num_trajs = config.n_traj

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  alg_name = config.alg_name
  run_name = f"{alg_name}_{config.tag}"
  wandb.init(project=env_name,
             name=run_name,
             entity='sangwon-seo',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  env = cb_env_factory(**env_kwargs)  # type: ParallelEnv

  # Seed envs
  env.reset(seed=seed)

  # bc_logger = sb3_logger.configure(log_dir, format_strs=("tensorboard", ))
  bc_logger = configure(log_dir, format_strings=("tensorboard", ))

  trajectories = load_trajectories(demo_path, num_trajs)

  list_transitions = []
  for aname in env.agents:
    agent_trajs = trajectories[aname]

    list_conv_trajs = []
    for i_e in range(len(agent_trajs["rewards"])):
      length = agent_trajs["lengths"][i_e]
      states = np.array(agent_trajs["states"][i_e])
      last_state = np.array(agent_trajs["next_states"][i_e][-1])
      states = np.concatenate(
          [states.reshape(length, -1),
           last_state.reshape(1, -1)], axis=0)
      done = agent_trajs["dones"][i_e][-1]
      actions = agent_trajs["actions"][i_e]
      list_conv_trajs.append(
          Trajectory(obs=states,
                     acts=np.array(actions).reshape(length, -1),
                     terminal=done,
                     infos=None))
    list_transitions.append(rollout.flatten_trajectories(list_conv_trajs))

  dummy_scheduler = lambda _: torch.finfo(torch.float32).max

  list_policy = []  # type: List[ActorCriticPolicy]
  list_bc_trainer = []  # type: List[BC]
  for aname in env.agents:
    list_policy.append(
        ActorCriticPolicy(observation_space=env.observation_space(aname),
                          action_space=env.action_space(aname),
                          lr_schedule=dummy_scheduler,
                          net_arch=dict(pi=config.hidden_policy,
                                        vf=config.hidden_critic),
                          activation_fn=nn.ReLU))
    list_bc_trainer.append(
        BC(observation_space=env.observation_space(aname),
           action_space=env.action_space(aname),
           policy=list_policy[aname],
           rng=np.random.default_rng(seed),
           demonstrations=list_transitions[aname],
           custom_logger=bc_logger,
           device="cpu",
           batch_size=config.mini_batch_size,
           optimizer_cls=torch.optim.Adam,
           optimizer_kwargs={'lr': config.optimizer_lr_policy}))

  best_eval_returns = -np.inf
  n_bc_trainer_iter = int(log_interval)
  n_iter = int(config.n_batches / n_bc_trainer_iter)
  # dict_eval_returns, eval_timesteps, wins = eval_policy(env, list_policy,
  #                                                       num_episodes)
  # ret_sum = np.zeros_like(dict_eval_returns[env.agents[0]])
  # for a_name in env.agents:
  #   ret_sum = ret_sum + np.array(dict_eval_returns[a_name])
  # mean_ret_sum = np.mean(ret_sum)
  # mean_wins = np.mean(wins)
  # mean_steps = np.mean(eval_timesteps)
  # print(f"Eval-0: reward {mean_ret_sum} |"
  #       f" step {mean_steps} | wins {mean_wins}")
  for idx in range(n_iter):
    for agent_idx in range(len(list_bc_trainer)):
      bc_trainer = list_bc_trainer[agent_idx]
      list_policy[agent_idx].set_training_mode(True)
      bc_trainer.train(n_batches=n_bc_trainer_iter,
                       log_interval=n_bc_trainer_iter + 1,
                       progress_bar=False)

    batch_so_far = (idx + 1) * n_bc_trainer_iter
    if batch_so_far % eval_interval == 0:
      dict_eval_returns, eval_timesteps, wins = eval_policy(
          env, list_policy, num_episodes)

      ret_sum = np.zeros_like(dict_eval_returns[env.agents[0]])
      for a_name in env.agents:
        ret_sum = ret_sum + np.array(dict_eval_returns[a_name])
        bc_logger.record(f'bc_eval/returns/{a_name}',
                         np.mean(dict_eval_returns[a_name]))

      mean_ret_sum = np.mean(ret_sum)
      mean_wins = np.mean(wins)
      mean_steps = np.mean(eval_timesteps)
      bc_logger.record('bc_eval/episode_reward', mean_ret_sum)
      bc_logger.record('bc_eval/episode_step', mean_steps)
      bc_logger.record('bc_eval/win_rate', mean_wins)
      bc_logger.dump(batch_so_far)
      print(f"Eval-{batch_so_far}: reward {mean_ret_sum} |"
            f" step {mean_steps} | wins {mean_wins}")

      if mean_ret_sum >= best_eval_returns:
        best_eval_returns = mean_ret_sum
        wandb.run.summary["best_returns"] = best_eval_returns

        if not os.path.exists(output_dir):
          os.mkdir(output_dir)

        for a_idx in range(len(list_policy)):
          file_path = os.path.join(output_dir,
                                   f'{env_name}_n{num_trajs}_l0_best_{a_idx}')
          list_policy[a_idx].save(file_path)

  wandb.finish()
