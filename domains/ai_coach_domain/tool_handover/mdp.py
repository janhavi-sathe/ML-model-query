import numpy as np
from ai_coach_core.models.mdp import LatentMDP
from ai_coach_core.utils.mdp_utils import StateSpace
from ai_coach_domain.tool_handover.transition import tool_handover_transition
import ai_coach_domain.tool_handover.define as tho


class MDP_ToolHandover(LatentMDP):

  def __init__(self, list_tool_types):
    self.list_tool_types = list_tool_types
    super().__init__(use_sparse=True)

  def init_statespace(self):

    self.statespace_patient_state = StateSpace(tho.PatientState)
    self.statespace_surgeon_sight = StateSpace(tho.SurgeonSight)
    self.statespace_surgical_step = StateSpace(tho.SurgicalStep)
    self.statespace_nurse_hand = StateSpace(tho.NURSE_HAND_POSITIONS)
    self.dict_factored_statespace = {
        0: self.statespace_patient_state,
        1: self.statespace_surgeon_sight,
        2: self.statespace_surgical_step,
        3: self.statespace_nurse_hand
    }

    self.idx_tool_start = len(self.dict_factored_statespace)

    for idx in range(len(self.list_tool_types)):
      self.dict_factored_statespace[self.idx_tool_start + idx] = StateSpace(
          tho.Tool_Location)

    self.dummy_states = StateSpace([tho.SURGICAL_STEP_TERMINAL])

  def init_actionspace(self):
    self.actionspace_surgeon_action = tho.SURGEON_ACTIONSPACE
    self.actionspace_nurse_action = tho.NURSE_ACTIONSPACE
    self.dict_factored_actionspace = {
        0: self.actionspace_surgeon_action,
        1: self.actionspace_nurse_action
    }

  def init_latentspace(self):
    self.latent_space = tho.MENTAL_STATESPACE

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    pat_state, sight, step, hand, tools = tup_states

    # terminal state
    if step == tho.SURGICAL_STEP_TERMINAL:
      return self.conv_dummy_state_to_idx(tho.SURGICAL_STEP_TERMINAL)

    pat_sidx = self.statespace_patient_state.state_to_idx[pat_state]
    sight_sidx = self.statespace_surgeon_sight.state_to_idx[sight]
    step_sidx = self.statespace_surgical_step.state_to_idx[step]
    hand_sidx = self.statespace_nurse_hand.state_to_idx[hand]

    list_sidx = [pat_sidx, sight_sidx, step_sidx, hand_sidx]
    for idx in range(len(tools)):
      list_sidx.append(
          self.dict_factored_statespace[self.idx_tool_start +
                                        idx].state_to_idx[tools[idx]])

    return self.conv_state_to_idx(tuple(list_sidx))

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    if self.is_dummy_state(state_idx):
      return None, None, self.conv_idx_to_dummy_state(state_idx), None, None

    state_vec = self.conv_idx_to_state(state_idx)
    pat = self.statespace_patient_state.idx_to_state[state_vec[0]]
    sight = self.statespace_surgeon_sight.idx_to_state[state_vec[1]]
    step = self.statespace_surgical_step.idx_to_state[state_vec[2]]
    hand = self.statespace_nurse_hand.idx_to_state[state_vec[3]]

    tools = []
    for idx in range(self.idx_tool_start, len(state_vec)):
      tools.append(
          self.dict_factored_statespace[idx].idx_to_state[state_vec[idx]])

    return pat, sight, step, hand, tools

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

    pat, sight, step, hand, tools = self.conv_mdp_sidx_to_sim_states(state_idx)
    surgeon_act, nurse_act = self.conv_mdp_aidx_to_sim_actions(action_idx)
    list_p_next_env = tool_handover_transition(pat, sight, step, hand,
                                               self.list_tool_types, tools,
                                               surgeon_act, nurse_act)

    map_next_state = {}
    for prop, s_pat, s_sight, s_step, s_hand, s_tool in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [s_pat, s_sight, s_step, s_hand, s_tool])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + prop

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

    pat, sight, step, hand, tools = self.conv_mdp_sidx_to_sim_states(state_idx)
    # if surgeon is looking at patient, they can't indicate a tool.
    if sight == tho.SurgeonSight.Patient:
      possible_actions = []
      for aidx in range(self.num_actions):
        surgeon_action, nurse_action = self.conv_mdp_aidx_to_sim_actions(aidx)
        if surgeon_action[0] != tho.SurgeonAction.Indicate_Tool:
          possible_actions.append(aidx)

      return possible_actions

    return super().legal_actions(state_idx)
