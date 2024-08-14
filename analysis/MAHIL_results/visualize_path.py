import os
import numpy as np
from omegaconf import OmegaConf
import cv2
from pettingzoo_domain.labor_division_v2 import (TwoTargetDyadLaborDivisionV2,
                                                 ThreeTargetDyadLaborDivisionV2,
                                                 LDv2Expert)

import env_loader
import itertools


def draw_triangle(canvas, pt, dir, color):
  if np.linalg.norm(dir) < 0.01:
    return canvas

  ortho = np.array([dir[1], -dir[0]])
  ortho = ortho / np.linalg.norm(ortho)
  len_dir = np.linalg.norm(dir)

  pt1 = pt + 0.5 * dir
  vec_ortho = len_dir * 0.15 * ortho
  pt2 = pt + vec_ortho
  pt3 = pt - vec_ortho
  pts = np.array([pt1, pt2, pt3])
  pts = np.int32(pts)
  canvas = cv2.fillPoly(canvas, [pts], color)
  return canvas


def draw_arrow(canvas, pt, dir, color):
  if np.linalg.norm(dir) < 0.01:
    return canvas

  ortho = np.array([dir[1], -dir[0]])
  ortho = ortho / np.linalg.norm(ortho)
  len_dir = np.linalg.norm(dir)

  pt1 = (pt + 0.5 * dir).astype(np.int32)
  ptm = pt + 0.3 * dir
  vec_ortho = len_dir * 0.2 * ortho
  pt2 = (ptm + vec_ortho).astype(np.int32)
  pt3 = (ptm - vec_ortho).astype(np.int32)

  thickness = 2
  canvas = cv2.line(canvas, pt, pt1, color, thickness)
  canvas = cv2.line(canvas, pt1, pt2, color, thickness)
  canvas = cv2.line(canvas, pt1, pt3, color, thickness)
  return canvas


def save_single_agent_path(env: ThreeTargetDyadLaborDivisionV2, agent_name,
                           agent, output_path, fixed_latent, n_epi, max_steps):
  cnvs_sz = env.unit_scr_sz * (env.world_high - env.world_low)
  canvas = np.ones((*cnvs_sz, 3), dtype=np.uint8) * 255
  canvas = env._draw_background(canvas)

  colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
  for i_e in range(n_epi):
    joint_obs, infos = env.reset()

    latent = fixed_latent

    done = False
    # while not done:
    for step in itertools.count():
      cur_pos = env.env_pt_2_scr_pt(joint_obs[agent_name][0:2])
      action = agent.choose_policy_action(joint_obs[agent_name],
                                          latent,
                                          sample=False)

      action_dir = (action * env.unit_scr_sz).astype(np.int64)

      joint_actions = {0: np.zeros(2), 1: np.zeros(2)}
      joint_actions[agent_name] = action

      joint_obs, rewards, dones, truncates, infos = env.step(joint_actions)

      if agent_name == 0:
        canvas = draw_triangle(canvas, cur_pos, action_dir, colors[latent])
      else:
        canvas = draw_arrow(canvas, cur_pos, action_dir, colors[latent])

      for a_name in env.agents:
        if dones[a_name] or truncates[a_name]:
          done = True
          break

      if done or step >= max_steps:
        break

  cv2.imwrite(output_path, canvas)


def iterate_saving_single_agent_paths_w_supervision(env_name, env, output_dir,
                                                    experts):
  supervision = 0.2
  model_num = 2
  alg_names = ["expert", "mahil", "maogail"]
  agent_idxs = [0, 1]
  latents = [0, 1, 2]

  for alg_name in alg_names:
    for agent_idx in agent_idxs:
      for latent in latents:
        output_path = os.path.join(
            output_dir, f"{env_name}_{alg_name}_a{agent_idx}_x{latent}.png")

        if alg_name == "expert":
          agent = experts[agent_idx]
        else:
          agent = env_loader.load_agent(env_name, alg_name, agent_idx,
                                        supervision, model_num)

        save_single_agent_path(env,
                               agent_idx,
                               agent,
                               output_path,
                               latent,
                               n_epi=10,
                               max_steps=99999)


def save_multi_agent_path(env: ThreeTargetDyadLaborDivisionV2, dict_agents,
                          output_path, max_steps):
  cnvs_sz = env.unit_scr_sz * (env.world_high - env.world_low)
  canvas = np.ones((*cnvs_sz, 3), dtype=np.uint8) * 255
  canvas = env._draw_background(canvas)

  colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

  n_epi = 1
  for i_e in range(n_epi):
    init_pos = {
        0: np.array([3, 2], dtype=np.float64),
        1: np.array([0, -4], dtype=np.float64)
    }
    joint_obs, infos = env.reset(options=init_pos)

    joint_prev_lat = {
        0: dict_agents[0].PREV_LATENT,
        1: dict_agents[1].PREV_LATENT
    }

    joint_latents = {0: 0, 1: 2}
    # for a_name in env.agents:
    #   latent = dict_agents[a_name].choose_mental_state(joint_obs[a_name],
    #                                                    joint_prev_lat[a_name],
    #                                                    None,
    #                                                    sample=False)
    #   joint_latents[a_name] = latent

    done = False
    # while not done:
    for step in itertools.count():
      joint_cur_pos = {}
      joint_action_dirs = {}
      joint_actions = {}
      for a_name in env.agents:
        latent = joint_latents[a_name]
        joint_cur_pos[a_name] = env.env_pt_2_scr_pt(joint_obs[a_name][0:2])
        action = dict_agents[a_name].choose_policy_action(joint_obs[a_name],
                                                          joint_latents[a_name],
                                                          sample=False)

        joint_actions[a_name] = action

        joint_action_dirs[a_name] = (action * env.unit_scr_sz).astype(np.int64)
        if a_name == 0:
          canvas = draw_triangle(canvas, joint_cur_pos[a_name],
                                 joint_action_dirs[a_name], colors[latent])
        else:
          canvas = draw_arrow(canvas, joint_cur_pos[a_name],
                              joint_action_dirs[a_name], colors[latent])

      joint_obs, rewards, dones, truncates, infos = env.step(joint_actions)

      joint_prev_lat = joint_latents
      joint_latents = {}
      for a_name in env.agents:
        joint_latents[a_name] = dict_agents[a_name].choose_mental_state(
            joint_obs[a_name], joint_prev_lat[a_name], None, sample=False)

      for a_name in env.agents:
        if dones[a_name] or truncates[a_name]:
          done = True
          break

      if done or step >= max_steps:
        break

  cv2.imwrite(output_path, canvas)


def iterate_saving_multi_agent_paths(env_name, env, output_dir, experts):
  model_num = 1
  supervision = 0.2
  alg_names = ["expert", "mahil", "maogail"]
  seeds = [0, 1, 2, 3, 4]

  for alg_name in alg_names:
    for seed in seeds:
      env.reset(seed=seed)
      output_path = os.path.join(output_dir,
                                 f"{env_name}_{alg_name}_s{seed}.png")

      if alg_name == "expert":
        dict_agents = experts
      else:
        dict_agents = {
            0: env_loader.load_agent(env_name, alg_name, 0, supervision,
                                     model_num),
            1: env_loader.load_agent(env_name, alg_name, 1, supervision,
                                     model_num)
        }
      save_multi_agent_path(env, dict_agents, output_path, 10)


def iterate_saving_single_agent_paths_unsupervised(env_name, env, output_dir,
                                                   experts, model_num, seed,
                                                   max_steps):
  supervision = 0.0
  alg_names = ["mahil", "maogail"]
  agent_idxs = [0, 1]
  latents = [0, 1, 2]

  for alg_name in alg_names:
    for agent_idx in agent_idxs:
      for latent in latents:
        output_path = os.path.join(
            output_dir,
            f"un_{env_name}_{alg_name}_a{agent_idx}_m{model_num}_x{latent}.png")

        if alg_name == "expert":
          agent = experts[agent_idx]
        else:
          agent = env_loader.load_agent(env_name, alg_name, agent_idx,
                                        supervision, model_num)
        env.reset(seed=seed)
        save_single_agent_path(env,
                               agent_idx,
                               agent,
                               output_path,
                               latent,
                               n_epi=10,
                               max_steps=max_steps)


if __name__ == "__main__":
  env_name = "LaborDivision3-v2"

  cur_dir = os.path.dirname(__file__)
  output_dir = os.path.join(cur_dir, "analysis_results")

  if env_name == "LaborDivision2-v2":
    env = TwoTargetDyadLaborDivisionV2(render_mode="human")
  elif env_name == "LaborDivision3-v2":
    env = ThreeTargetDyadLaborDivisionV2(render_mode="human")
    experts = {
        0: LDv2Expert(env, env.tolerance, 0),
        1: LDv2Expert(env, env.tolerance, 1)
    }
  else:
    raise NotImplementedError()

  # iterate_saving_single_agent_paths_w_supervision(env_name, env, output_dir,
  #                                                 experts)

  # iterate_saving_multi_agent_paths(env_name,  env, output_dir,
  #                                  experts)

  iterate_saving_single_agent_paths_unsupervised(env_name,
                                                 env,
                                                 output_dir,
                                                 experts,
                                                 model_num=1,
                                                 seed=1,
                                                 max_steps=20)
