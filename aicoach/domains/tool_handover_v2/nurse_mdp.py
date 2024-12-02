import os
from typing import Optional, Sequence
import numpy as np
from TMM.models.mdp import LatentMDP, StateSpace, ActionSpace
from .transition import tool_handover_transition_v2
import aicoach.domains.tool_handover_v2.define as tho
from TMM.models.policy import CachedPolicyInterface, PolicyInterface
from TMM.models.agent_model import AgentModel

TERMINAL_STATE = "TERMINAL"
policy_list = []


class MDP_THO_Nurse(LatentMDP):

  def __init__(self, surgeon_pos, nurse_init_pos, nurse_init_dir,
               nurse_possible_pos, table_blocks, surgical_steps,
               tool_table_zone, **kwargs):
    self.surgeon_pos = surgeon_pos
    self.nurse_init_pos = nurse_init_pos
    self.nurse_init_dir = nurse_init_dir
    self.nurse_possible_pos = nurse_possible_pos
    self.table_blocks = table_blocks
    self.surgical_steps = surgical_steps
    self.tool_table_zone = tool_table_zone

    super().__init__(use_sparse=True)

  def init_statespace(self):

    self.statespace_nurse_dir = StateSpace(tho.NurseDirection)
    self.statespace_nurse_pos = StateSpace(self.nurse_possible_pos)
    self.statespace_nurse_tool = StateSpace(tho.POSSIBLE_TOOLS)
    self.statespace_surgeon_tool = StateSpace(tho.POSSIBLE_TOOLS)
    self.statespace_nurse_asked = StateSpace([True, False])

    self.dict_factored_statespace = {
        0: self.statespace_nurse_dir,
        1: self.statespace_nurse_pos,
        2: self.statespace_nurse_tool,
        3: self.statespace_surgeon_tool,
        4: self.statespace_nurse_asked
    }

    self.dummy_states = None

  def init_actionspace(self):
    self.actionspace_nurse_action = tho.NURSE_ACTIONSPACE

    self.dict_factored_actionspace = {0: self.actionspace_nurse_action}

  def init_latentspace(self):
    self.latent_space = StateSpace(tho.Requirement)

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    'sim_state: dir, pos, nurse_tool, surgeon_tool, nurse_asked'
    list_sidx = []
    for idx in range(len(tup_states)):
      list_sidx.append(
          self.dict_factored_statespace[idx].state_to_idx[tup_states[idx]])

    return self.conv_state_to_idx(tuple(list_sidx))

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    state_vec = self.conv_idx_to_state(state_idx)
    list_states = []
    for idx in range(len(state_vec)):
      list_states.append(
          self.dict_factored_statespace[idx].idx_to_state[state_vec[idx]])

    return tuple(list_states)

  def conv_sim_actions_to_mdp_aidx(self, tuple_actions):
    list_aidx = []
    for idx, act in enumerate(tuple_actions):
      list_aidx.append(self.dict_factored_actionspace[idx].action_to_idx[act])

    return self.np_action_to_idx[tuple(list_aidx)]

  def conv_mdp_aidx_to_sim_actions(self, action_idx):
    vector_aidx = self.conv_idx_to_action(action_idx)
    list_actions = []
    for idx, aidx in enumerate(vector_aidx):
      list_actions.append(
          self.dict_factored_actionspace[idx].idx_to_action[aidx])

    return tuple(list_actions)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    (nurs_dir, nurs_pos, nurs_tool, surg_tool,
     nurs_asked) = self.conv_mdp_sidx_to_sim_states(state_idx)
    nurs_action, = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = tool_handover_transition_v2(
        tho.PatientVital.Stable, nurs_dir, nurs_pos, nurs_tool, surg_tool,
        False, False, False, 0, tho.Requirement.Nurse_Assist, nurs_asked,
        nurs_action, tho.SurgeonAction.Stay, tho.AnesthesiaAction.Stay,
        tho.PerfusionAction.Stay, self.surgeon_pos, self.nurse_possible_pos,
        self.table_blocks, self.surgical_steps, self.tool_table_zone)

    map_next_state = {}
    for item in list_p_next_env:
      (_, s_n_dir, s_n_pos, s_n_tool, s_s_tool, _, _, _, _, _,
       s_n_ask) = item[1:]
      nurse_obs = (s_n_dir, s_n_pos, s_n_tool, s_s_tool, s_n_ask)

      sidx_n = self.conv_sim_states_to_mdp_sidx(nurse_obs)
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + item[0]

    list_next_p_state = []
    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def is_terminal(self, state_idx: int):
    return False

  def legal_actions(self, state_idx: int):
    if self.is_terminal(state_idx):
      return []

    tup_states = self.conv_mdp_sidx_to_sim_states(state_idx)
    (n_dir, n_pos, n_tool, s_tool, n_ask) = tup_states

    illegal_n_actions = []

    target_pos = tho.get_target_pos(n_pos, n_dir)

    if target_pos not in self.nurse_possible_pos:
      illegal_n_actions.append((tho.NurseAction.Move_Forward, None))

    if not tho.can_exchange_tool(n_pos, n_dir, self.surgeon_pos):
      illegal_n_actions.append((tho.NurseAction.Assist, None))

    if target_pos not in self.table_blocks:
      illegal_n_actions.append(
          (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant1))
      illegal_n_actions.append(
          (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant2))
      illegal_n_actions.append(
          (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant3))
      illegal_n_actions.append(
          (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant4))
    else:
      idx = self.table_blocks.index(target_pos)
      if n_tool == tho.Requirement.Hand_Only:  # pick up
        check_quad = [False, False, False, False]
        for tool, table_zone in self.tool_table_zone.items():
          if table_zone[0] == idx:
            check_quad[table_zone[1] - 1] = True
        for idx, check in enumerate(check_quad):
          if not check:
            illegal_n_actions.append(
                (tho.NurseAction.PickUp_Drop, tho.PickupLocation(idx + 1)))
      else:  # drop
        orig_zone = self.tool_table_zone[n_tool]
        for pick_loc in tho.PickupLocation:
          if orig_zone != (idx, pick_loc.value):
            illegal_n_actions.append((tho.NurseAction.PickUp_Drop, pick_loc))

    possible_actions = []
    for aidx in range(self.num_actions):
      n_act, = self.conv_mdp_aidx_to_sim_actions(aidx)
      if (n_act in illegal_n_actions):
        continue
      possible_actions.append(aidx)

    return possible_actions

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    req = self.latent_space.idx_to_state[latent_idx]
    tup_states = self.conv_mdp_sidx_to_sim_states(state_idx)
    (n_dir, n_pos, n_tool, s_tool, n_ask) = tup_states

    nurs_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)

    reward = -1
    if req == tho.Requirement.Nurse_Assist:
      if tho.can_exchange_tool(n_pos, n_dir, self.surgeon_pos):
        if nurs_act[0] == tho.NurseAction.Assist:
          reward += 1
    elif req == n_tool:
      if tho.can_exchange_tool(n_pos, n_dir, self.surgeon_pos):
        if nurs_act[0] == tho.NurseAction.Stay:
          reward += 1

    return reward


class THONursePolicy(CachedPolicyInterface):

  def __init__(self, mdp: MDP_THO_Nurse, temperature: float) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_nurse_")

    str_fileprefix += str(
        str(mdp.surgeon_pos) + str(mdp.nurse_init_pos) +
        str(mdp.nurse_init_dir.value) + str(len(mdp.nurse_possible_pos)) +
        str(len(mdp.table_blocks)) + str(len(mdp.surgical_steps)) +
        str(len(mdp.tool_table_zone))) + "_"

    super().__init__(mdp, str_fileprefix, policy_list, temperature, (0, ))
