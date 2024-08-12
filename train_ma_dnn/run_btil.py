import numpy as np
import itertools
from aic_core.utils.mdp_utils import StateSpace, ActionSpace
from aic_domain.box_push_v2 import AGENT_ACTIONSPACE, BoxState
from pettingzoo_domain.po_movers_v2 import PO_Movers_V2


class Converter_PO_Movers:

  def __init__(self) -> None:
    self.num_latent = 4
    self.env = PO_Movers_V2(render_mode=None)

    self.init_statespace()
    self.init_statespace_helper_vars()

  def init_statespace(self):
    self.dict_factored_statespace = {}

    possible_coords = [(x, y) for x in range(7) for y in range(7)]

    self.my_pos_space = StateSpace(possible_coords)

    action_idxs = list(range(AGENT_ACTIONSPACE.num_actions))
    self.my_action_space = StateSpace(action_idxs)

    possible_otherplayer_state = [None]
    for x in [-1, 0, 1]:
      for y in [-1, 0, 1]:
        for act in action_idxs:
          possible_otherplayer_state.append((x, y, act))
    self.otherplayer_space = StateSpace(possible_otherplayer_state)

    box_states = []
    for item in itertools.product(range(4), repeat=3):
      box_states.append(item)

    self.box_space = StateSpace(box_states)

    self.dict_factored_statespace = {
        0: self.my_pos_space,
        1: self.my_action_space,
        2: self.otherplayer_space,
        3: self.box_space
    }

    self.dummy_states = None

  def init_statespace_helper_vars(self):

    # Retrieve number of states and state factors.
    self.num_state_factors = len(self.dict_factored_statespace)
    self.list_num_states = []
    for idx in range(self.num_state_factors):
      self.list_num_states.append(
          self.dict_factored_statespace.get(idx).num_states)

    self.num_actual_states = np.prod(self.list_num_states)
    self.num_dummy_states = (0 if self.dummy_states is None else
                             self.dummy_states.num_states)
    self.num_states = self.num_actual_states + self.num_dummy_states

    # Create mapping from state to state index.
    # Mapping takes state value as inputs and outputs a scalar state index.
    np_list_idx = np.arange(self.num_actual_states, dtype=np.int32)
    self.np_state_to_idx = np_list_idx.reshape(self.list_num_states)

    # Create mapping from state index to state.
    # Mapping takes state index as input and outputs a factored state.
    np_idx_to_state = np.zeros((self.num_actual_states, self.num_state_factors),
                               dtype=np.int32)
    for state, idx in np.ndenumerate(self.np_state_to_idx):
      np_idx_to_state[idx] = state
    self.np_idx_to_state = np_idx_to_state

  def convert_obs_2_sidx(self, obs):
    my_x = np.where(obs[0:7] == 1)[0][0]
    my_y = np.where(obs[7:14] == 1)[0][0]
    my_a = np.where(obs[14:20] == 1)[0][0]

    fr_o = bool(obs[20])
    fr_x = np.where(obs[21:24] == 1)[0][0] - 1
    fr_y = np.where(obs[24:27] == 1)[0][0] - 1
    fr_a = np.where(obs[27:33] == 1)[0][0]

    box1 = np.where(obs[33:37] == 1)[0][0]
    box2 = np.where(obs[37:41] == 1)[0][0]
    box3 = np.where(obs[41:45] == 1)[0][0]

    mypos_idx = self.my_pos_space.state_to_idx[(my_x, my_y)]
    myact_idx = self.my_action_space.state_to_idx[my_a]
    if fr_o:
      otherplayer_idx = self.otherplayer_space.state_to_idx[(fr_x, fr_y, fr_a)]
    else:
      otherplayer_idx = self.otherplayer_space.state_to_idx[None]

    box_idx = self.box_space.state_to_idx[(box1, box2, box3)]

    sidx = self.np_state_to_idx[(mypos_idx, myact_idx, otherplayer_idx,
                                 box_idx)]
    return sidx

  def convert_trajectories(self, trajectories):

    trajectories["states"][0]
    pass
