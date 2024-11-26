import numpy as np
import os
from TMM.models.mdp import LatentMDP, StateSpace, ActionSpace
from .transition import tool_handover_transition_v2
import aicoach.domains.tool_handover_v2.define as tho

TERMINAL_STATE = "TERMINAL"


class MDP_ToolHandover_V2(LatentMDP):

  def __init__(self, surgeon_pos, patient_pos_size, perf_pos, anes_pos,
               nurse_init_pos, nurse_init_dir, nurse_possible_pos, table_blocks,
               vital_pos, surgical_steps, tool_table_zone, width, height,
               **kwargs):
    self.width = width
    self.height = height
    self.surgeon_pos = surgeon_pos
    self.patient_pos_size = patient_pos_size
    self.perf_pos = perf_pos
    self.anes_pos = anes_pos
    self.nurse_init_pos = nurse_init_pos
    self.nurse_init_dir = nurse_init_dir
    self.nurse_possible_pos = nurse_possible_pos
    self.table_blocks = table_blocks
    self.vital_pos = vital_pos
    self.surgical_steps = surgical_steps
    self.tool_table_zone = tool_table_zone

    cur_dir = os.path.dirname(__file__)
    str_cache_path = os.path.join(cur_dir, "data/")

    super().__init__(use_sparse=True, cache_file_path=str_cache_path)

  def init_statespace(self):

    self.statespace_patient_vital = StateSpace(tho.PatientVital)
    self.statespace_nurse_dir = StateSpace(tho.NurseDirection)
    self.statespace_nurse_pos = StateSpace(self.nurse_possible_pos)
    self.statespace_nurse_tool = StateSpace(tho.POSSIBLE_TOOLS)
    self.statespace_surgeon_tool = StateSpace(tho.POSSIBLE_TOOLS)
    self.statespace_surgeon_ready = StateSpace([True, False])
    self.statespace_anes_ready = StateSpace([True, False])
    self.statespace_perf_ready = StateSpace([True, False])
    self.statespace_cur_step = StateSpace(range(len(self.surgical_steps)))
    self.statespace_cur_requirement = StateSpace(tho.Requirement)
    self.statespace_nurse_asked = StateSpace([True, False])

    self.dict_factored_statespace = {
        0: self.statespace_patient_vital,
        1: self.statespace_nurse_dir,
        2: self.statespace_nurse_pos,
        3: self.statespace_nurse_tool,
        4: self.statespace_surgeon_tool,
        5: self.statespace_surgeon_ready,
        6: self.statespace_anes_ready,
        7: self.statespace_perf_ready,
        8: self.statespace_cur_step,
        9: self.statespace_cur_requirement,
        10: self.statespace_nurse_asked
    }

    self.dummy_states = StateSpace([TERMINAL_STATE])

  def init_actionspace(self):
    self.actionspace_nurse_action = tho.NURSE_ACTIONSPACE
    self.actionspace_surgeon_action = ActionSpace(tho.SurgeonAction)
    self.actionspace_anes_action = ActionSpace(tho.AnesthesiaAction)
    self.actionspace_perf_action = ActionSpace(tho.PerfusionAction)

    self.dict_factored_actionspace = {
        0: self.actionspace_nurse_action,
        1: self.actionspace_surgeon_action,
        2: self.actionspace_anes_action,
        3: self.actionspace_perf_action
    }

  def init_latentspace(self):
    self.latent_space = StateSpace(tho.Requirement)

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    # terminal state
    if tup_states[8] >= len(self.surgical_steps):
      return self.conv_dummy_state_to_idx(TERMINAL_STATE)

    list_sidx = []
    for idx in range(len(tup_states)):
      list_sidx.append(
          self.dict_factored_statespace[idx].state_to_idx[tup_states[idx]])

    return self.conv_state_to_idx(tuple(list_sidx))

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    if self.is_dummy_state(state_idx):
      list_dummy = []
      for idx in range(len(self.dict_factored_statespace)):
        list_dummy.append(self.dict_factored_statespace[idx].idx_to_state[0])
      list_dummy[8] = len(self.surgical_steps)

      return tuple(list_dummy)

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

    tup_states = self.conv_mdp_sidx_to_sim_states(state_idx)
    tup_actions = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = tool_handover_transition_v2(
        *tup_states, *tup_actions, self.surgeon_pos, self.nurse_possible_pos,
        self.table_blocks, self.surgical_steps, self.tool_table_zone)

    map_next_state = {}
    for item in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(item[1:])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + item[0]

    list_next_p_state = []
    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def is_terminal(self, state_idx: int):
    return self.is_dummy_state(state_idx)

  def legal_actions(self, state_idx: int):
    if self.is_terminal(state_idx):
      return []

    tup_states = self.conv_mdp_sidx_to_sim_states(state_idx)
    (vital, n_dir, n_pos, n_tool, s_tool, s_rdy, a_rdy, p_rdy, step, req,
     n_ask) = tup_states

    illegal_n_actions = []
    illegal_s_actions = []
    illegal_a_actions = []
    illegal_p_actions = []

    if not tho.can_exchange_tool(n_pos, n_dir, self.surgeon_pos):
      illegal_s_actions.append(tho.SurgeonAction.Exchange_Tool)
      illegal_n_actions.append((tho.NurseAction.Assist, None))
    # elif req != tho.Requirement.Nurse_Assist:
    #   illegal_n_actions.append((tho.NurseAction.Assist, None))

    if (req != tho.Requirement.Nurse_Assist and s_tool != req) or s_rdy:
      illegal_s_actions.append(tho.SurgeonAction.Proceed)

    if p_rdy:
      illegal_p_actions.append(tho.PerfusionAction.Proceed)

    if a_rdy:
      illegal_a_actions.append(tho.AnesthesiaAction.Proceed)

    target_pos = tho.get_target_pos(n_pos, n_dir)

    if target_pos not in self.nurse_possible_pos:
      illegal_n_actions.append((tho.NurseAction.Move_Forward, None))

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
      n_act, s_act, a_act, p_act = self.conv_mdp_aidx_to_sim_actions(aidx)
      if (n_act in illegal_n_actions or s_act in illegal_s_actions
          or a_act in illegal_a_actions or p_act in illegal_p_actions):
        continue
      possible_actions.append(aidx)

    return possible_actions
