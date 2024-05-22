import numpy as np
import torch
from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from omegaconf import OmegaConf, DictConfig


def parse_smacv2_distribution(n_allies, n_enemies, map_name):
  distribution_config = {
      "n_units": int(n_allies),
      "n_enemies": int(n_enemies),
      "start_positions": {
          "dist_type": "surrounded_and_reflect",
          "p": 0.5,
          "map_x": 32,
          "map_y": 32,
      }
  }
  if 'protoss' in map_name:
    distribution_config['team_gen'] = {
        "dist_type": "weighted_teams",
        "unit_types": ["stalker", "zealot", "colossus"],
        "weights": [0.45, 0.45, 0.1],
        "observe": True,
    }
  elif 'zerg' in map_name:
    distribution_config['team_gen'] = {
        "dist_type": "weighted_teams",
        "unit_types": ["zergling", "baneling", "hydralisk"],
        "weights": [0.45, 0.1, 0.45],
        "observe": True,
    }
  elif 'terran' in map_name:
    distribution_config['team_gen'] = {
        "dist_type": "weighted_teams",
        "unit_types": ["marine", "marauder", "medivac"],
        "weights": [0.45, 0.45, 0.1],
        "observe": True,
    }
  return distribution_config


def restore(actor, model_dir):
  """Restore policy's networks from a saved model."""
  policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
  actor.load_state_dict(policy_actor_state_dict)


if __name__ == "__main__":
  args = DictConfig({})
  # args
  args["hidden_size"] = 64
  args["gain"] = 0.01
  args["use_orthogonal"] = True
  args["use_policy_active_masks"] = True
  args["algorithm_name"] = "mappo"
  if args["algorithm_name"] == "rmappo":
    args["use_recurrent_policy"] = True
    args["use_naive_recurrent_policy"] = False
  elif args["algorithm_name"] == "mappo":
    args["use_recurrent_policy"] = False
    args["use_naive_recurrent_policy"] = False
  args["recurrent_N"] = 1
  args["use_feature_normalization"] = True
  args["use_ReLU"] = True
  args["stacked_frames"] = 1
  args["layer_N"] = 1
  n_allies, n_enemies, map_name = 5, 5, '10gen_protoss'
  episode_length = 400
  n_traj = 50
  device = "cpu"
  model_dir = "/home/sangwon/Projects/ai_coach/train_ma_dnn/StarCraft2v2_res/StarCraft2v2/10gen_protoss/mappo/protoss/run1/models"

  distribution_config = parse_smacv2_distribution(n_allies, n_enemies, map_name)
  env = SMACv2(capability_config=distribution_config, map_name=map_name)

  # load model
  actor = R_Actor(args, env.observation_space[0], env.action_space[0], device)
  restore(actor, model_dir)
  actor.eval()
  # actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)

  deterministic = True
  num_win = 0
  list_length = []

  dummy_value = np.zeros(2)

  for _ in range(n_traj):
    obs, share_obs, available_actions = env.reset()
    for step in range(episode_length):
      # Sample actions
      with torch.no_grad():
        obs = np.array(obs)
        available_actions = np.array(available_actions)
        actions, _, _ = actor(obs, dummy_value, dummy_value, available_actions,
                              deterministic)

      # Obser reward and next obs
      obs, share_obs, rewards, dones, infos, available_actions = env.step(
          np.array(actions))

      trunc = False
      done = np.all(dones)
      if done:
        if infos[0]['won']:
          num_win += 1
        break
      elif step == episode_length - 1:
        trunc = True
        break
    list_length.append(step)

  win_rate = num_win / n_traj
  print("========= win_rate:", win_rate)
  print("========= avg traj len:", sum(list_length) / len(list_length))
  print(max(list_length))
