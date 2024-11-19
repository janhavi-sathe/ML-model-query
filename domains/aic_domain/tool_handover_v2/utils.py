from typing import Callable, Tuple
from aic_domain.tool_handover_v2.mdp import MDP_ToolHandover_V2
from aic_domain.tool_handover_v2.simulator import ToolHandoverV2Simulator
import numpy as np
from aic_core.utils.data_utils import Trajectories
from aic_domain.rescue_v2.define import AGENT_ACTIONSPACE
from aic_domain.rescue_v2.simulator import RescueSimulatorV2
from aic_domain.rescue_v2.mdp import MDP_Rescue_Task

from aic_domain.tool_handover_v2.define import NURSE_ACTIONSPACE
from aic_domain.rescue_v2.simulator import RescueSimulatorV2
from aic_domain.rescue_v2.mdp import MDP_Rescue_Task


class ToolHandoverV2Trajectories(Trajectories):

  def __init__(self, task_mdp: MDP_ToolHandover_V2, tup_num_latents: Tuple[int,
                                                                       ...],
               cb_conv_latent_to_idx: Callable[[int, int], int]) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=4,
                     num_latent_factors=1,
                     tup_num_latents=tup_num_latents)
    self.task_mdp = task_mdp
    self.cb_conv_latent_to_idx = cb_conv_latent_to_idx

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = ToolHandoverV2Simulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        (state, actions, latent) = vec_state_action

        sidx = self.task_mdp.conv_sim_states_to_mdp_sidx(state)
        aidxn = NURSE_ACTIONSPACE.action_to_idx[actions[0]] if actions[0] is not None else Trajectories.EPISODE_END
        aidxs = self.task_mdp.dict_factored_actionspace[1].action_to_idx[actions[1]] if actions[1] is not None else Trajectories.EPISODE_END
        aidxa = self.task_mdp.dict_factored_actionspace[2].action_to_idx[actions[2]] if actions[2] is not None else Trajectories.EPISODE_END
        aidxp = self.task_mdp.dict_factored_actionspace[3].action_to_idx[actions[3]] if actions[3] is not None else Trajectories.EPISODE_END

        xidx1 = (self.cb_conv_latent_to_idx(latent)
                 if latent is not None else Trajectories.EPISODE_END)
       
        np_trj[tidx, :] = [sidx, aidxn, aidxs, aidxa, aidxp, xidx1]

      self.list_np_trajectory.append(np_trj)
