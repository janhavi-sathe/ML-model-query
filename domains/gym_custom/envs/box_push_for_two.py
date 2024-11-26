from typing import Optional, Sequence
from gym_custom.envs.mdp_env.env_from_mdp import EnvFromMDP
from aic_core.models.mdp import MDP
from aic_domain.box_push.maps import EXP1_MAP
from aic_domain.box_push.policy import (BoxPushPolicyTeamExp1,
                                        BoxPushPolicyIndvExp1)
from aic_domain.box_push.simulator import (BoxPushSimulator_AlwaysTogether,
                                           BoxPushSimulator_AlwaysAlone)
from aic_domain.box_push.mdp import (BoxPushTeamMDP_AlwaysTogether,
                                     BoxPushTeamMDP_AlwaysAlone,
                                     BoxPushAgentMDP_AlwaysAlone)
from aic_domain.box_push.agent import (BoxPushAIAgent_Indv2,
                                       BoxPushAIAgent_Team2)
from gym import spaces
import numpy as np

TEMPERATURE = 0.3

if __name__ == "__main__":
  from aic_domain.box_push.utils import BoxPushTrajectories
  from collections import defaultdict
  import numpy as np
  import os
  import pickle
  import glob

  cur_dir = os.path.dirname(__file__)

  def conv_human_data_2_iql_format(task_mdp, agent_mdp, load_files, save_dir,
                                   env_name):
    train_data = BoxPushTrajectories(task_mdp, agent_mdp)
    train_data.load_from_files(load_files)
    list_trajectories = train_data.get_as_column_lists(include_terminal=True)

    expert_trajs = defaultdict(list)
    for s_array, a_array, x_array in list_trajectories:

      expert_trajs["states"].append(s_array[:-1])
      expert_trajs["next_states"].append(s_array[1:])

      actions, _ = list(zip(*a_array))
      latents, _ = list(zip(*x_array))

      expert_trajs["actions"].append(actions)
      expert_trajs["latents"].append(latents)

      leng = len(a_array)

      expert_trajs["lengths"].append(leng)
      expert_trajs["rewards"].append(-1)

      dones = [False] * leng
      dones[-1] = task_mdp.is_terminal(s_array[-1])

      expert_trajs["dones"].append(dones)

    num_data = len(expert_trajs["states"])
    if save_dir is not None:
      save_path = os.path.join(save_dir, f"{env_name}_{num_data}.pkl")
      with open(save_path, 'wb') as f:
        pickle.dump(expert_trajs, f)

    return expert_trajs

  DATA_DIR = "/home/sangwon/Projects/ai_coach/analysis/BTIL_results/aws_data_test/"

  # env_movers = EnvMovers_v0()
  # print(env_movers.mdp.num_latents)

  # movers_data = glob.glob(os.path.join(DATA_DIR + "domain1", '*.txt'))
  # # traj = conv_human_data_2_iql_format(
  # #     env_movers.mdp, env_movers.robot_agent.agent_model.get_reference_mdp(),
  # #     movers_data[:1], None, "EnvMovers_v0")
  # num_train = 44
  # conv_human_data_2_iql_format(
  #     env_movers.mdp, env_movers.robot_agent.agent_model.get_reference_mdp(),
  #     movers_data, cur_dir, "EnvMovers_v0")
  # conv_human_data_2_iql_format(
  #     env_movers.mdp, env_movers.robot_agent.agent_model.get_reference_mdp(),
  #     movers_data[:num_train], cur_dir, "EnvMovers_v0")
  # conv_human_data_2_iql_format(
  #     env_movers.mdp, env_movers.robot_agent.agent_model.get_reference_mdp(),
  #     movers_data[num_train:], cur_dir, "EnvMovers_v0")

  env_cleanup = EnvCleanup_v0()
  print(env_cleanup.mdp.num_latents)
  cleanup_data = glob.glob(os.path.join(DATA_DIR + "domain2", '*.txt'))
  num_train = 66
  conv_human_data_2_iql_format(
      env_cleanup.mdp, env_cleanup.robot_agent.agent_model.get_reference_mdp(),
      cleanup_data, cur_dir, "EnvCleanup_v0")
  # conv_human_data_2_iql_format(
  #     env_cleanup.mdp, env_cleanup.robot_agent.agent_model.get_reference_mdp(),
  #     cleanup_data[:num_train], cur_dir, "EnvCleanup_v0")
  # conv_human_data_2_iql_format(
  #     env_cleanup.mdp, env_cleanup.robot_agent.agent_model.get_reference_mdp(),
  #     cleanup_data[num_train:], cur_dir, "EnvCleanup_v0")
