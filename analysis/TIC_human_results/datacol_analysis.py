import os
import glob
import numpy as np
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.box_push_v2.mdp import MDP_Movers_Agent, MDP_Movers_Task
from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
from aic_domain.box_push.utils import BoxPushTrajectories
from aic_domain.rescue.utils import RescueTrajectories

if __name__ == "__main__":
  dir_name = "/home/sangwon/Projects/ai_coach/analysis/TIC_results/human_data/"

  domain = "flood"

  if domain == "movers":
    game_map = MAP_MOVERS
    mdp_task = MDP_Movers_Task(**game_map)
    mdp_agent = MDP_Movers_Agent(**game_map)
    train_data = BoxPushTrajectories(mdp_task, mdp_agent)
    possible_latents = [
        mdp_agent.latent_space.idx_to_state[i]
        for i in range(mdp_agent.num_latents)
    ]
  elif domain == "flood":
    game_map = MAP_RESCUE
    mdp_task = MDP_Rescue_Task(**game_map)
    mdp_agent = MDP_Rescue_Agent(**game_map)

    def conv_latent_to_idx(agent_idx, latent):
      return mdp_agent.latent_space.state_to_idx[latent]

    train_data = RescueTrajectories(
        mdp_task, (mdp_agent.num_latents, mdp_agent.num_latents),
        conv_latent_to_idx)
    possible_latents = [
        game_map["places"][game_map["work_locations"][i].id].name
        for i in range(mdp_agent.num_latents)
    ]

  train_dir = os.path.join(dir_name, game_map["name"] + '_train')

  # load train set
  ##################################################
  file_names = glob.glob(os.path.join(train_dir, '*.txt'))

  # dict_list_per_latent = {mdp_agent.latent_space.idx_to_state[i]: [] for i in range(mdp_agent.num_latents)}
  list_latent_counts = []
  train_data.load_from_files(file_names)
  traj_labeled_ver = train_data.get_as_column_lists(True)
  for _, _, latents in traj_labeled_ver:
    counts = [0] * mdp_agent.num_latents
    for x1, _ in latents:
      counts[x1] += 1

    list_latent_counts.append(counts)

  np_latent_counts = np.array(list_latent_counts)

  print(possible_latents)
  print(np.mean(np_latent_counts, axis=0))
  print(np.median(np_latent_counts, axis=0))
  print(np.quantile(np_latent_counts, 0.25, axis=0))
  print(np.quantile(np_latent_counts, 0.75, axis=0))
