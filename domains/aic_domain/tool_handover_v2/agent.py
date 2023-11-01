from typing import Union
from aic_core.models.agent_model import AgentModel
from aic_core.models.policy import CachedPolicyInterface
from aic_domain.agent import SimulatorAgent, AIAgent_PartialObs
import aic_domain.tool_handover_v2.define as tho
import random
import numpy as np
from aic_domain.tool_handover_v2.surgery_info import CABG_INFO
from aic_domain.tool_handover_v2.nurse_mdp import THONursePolicy


class MindlessAgent(SimulatorAgent):

  def __init__(self) -> None:
    super().__init__(False, True)
    self.manual_action = None

  def init_latent(self, tup_states):
    pass

  def get_current_latent(self):
    return None

  def set_latent(self, latent):
    pass

  def set_action(self, action):
    self.manual_action = action

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    pass


class AnesthesiaAgent(MindlessAgent):

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    if tup_states[6]:
      return tho.AnesthesiaAction.Stay
    else:
      P_PROCEED = 0.9
      if random.random() < P_PROCEED:
        return tho.AnesthesiaAction.Proceed
      else:
        return tho.AnesthesiaAction.Stay


class PerfusionAgent(MindlessAgent):

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    if tup_states[7]:
      return tho.PerfusionAction.Stay
    else:
      P_PROCEED = 0.9
      if random.random() < P_PROCEED:
        return tho.PerfusionAction.Proceed
      else:
        return tho.PerfusionAction.Stay


class SurgeonAgent(MindlessAgent):

  def __init__(self, surgeon_pos) -> None:
    super().__init__()
    self.surgeon_pos = surgeon_pos

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    (vital, n_dir, n_pos, n_tool, s_tool, s_rdy, a_rdy, p_rdy, step, req,
     n_ask) = tup_states

    if n_ask:
      return tho.SurgeonAction.Tell_Requirement

    if s_rdy:
      return tho.SurgeonAction.Stay

    if req == tho.Requirement.Nurse_Assist:
      return tho.SurgeonAction.Proceed
    elif s_tool == req:
      return tho.SurgeonAction.Proceed
    else:
      if tho.can_exchange_tool(n_pos, n_dir, self.surgeon_pos):
        if n_tool == req or n_tool == tho.Requirement.Hand_Only:
          return tho.SurgeonAction.Exchange_Tool
        else:
          P_TELL = 0.2
          if random.random() < P_TELL:
            return tho.SurgeonAction.Tell_Requirement
          else:
            return tho.SurgeonAction.Stay
      else:
        return tho.SurgeonAction.Stay


# TODO: refactoring - consider merging with AIAgent_PartialObs
class NurseAgent(SimulatorAgent):

  def __init__(self, policy_model: THONursePolicy, experienced: bool) -> None:
    super().__init__(True, True)

    self.policy_model = policy_model
    self.cur_latent = None  # type: Union[tho.Requirement, None]
    self.manual_action = None
    self.assumed_tup_states = None
    self.experienced = experienced
    self.surgical_steps = CABG_INFO['surgical_steps']

  def observed_action_state(self, tup_actions, tup_nxt_state):
    # TODO: implement this
    (patient_vital, nurse_dir, nurse_pos, nurse_tool, surgeon_tool,
     surgeon_ready, perf_ready, anes_ready, cur_step, cur_requirement,
     nurse_asked) = tup_nxt_state

    obs_requirement = None
    if tup_actions is not None:
      nurs_act, surg_act, anes_act, perf_act = tup_actions
      obs_action = (nurs_act, surg_act)

      if surg_act == tho.SurgeonAction.Tell_Requirement:
        obs_requirement = cur_requirement
    else:
      obs_action = None

    if not self.experienced:
      cur_step = None

    obs_state = (nurse_dir, nurse_pos, nurse_tool, surgeon_tool, nurse_asked,
                 cur_step, obs_requirement)
    return obs_action, obs_state

  def sample_latent_from_surgical_info_dict(self, cur_surgical_step):
    next_req_prop = (
        self.surgical_steps[cur_surgical_step][tho.PatientVital.Stable])

    next_reqs = []
    next_props = []
    for next_req, req_prop in next_req_prop:
      next_reqs.append(next_req)
      next_props.append(req_prop)

    return np.random.choice(next_reqs, p=next_props)

  def init_latent(self, tup_states):
    self.assumed_tup_states = self.observed_action_state(None, tup_states)[1]
    if not self.experienced:
      req = random.choice(list(tho.Requirement))
      self.cur_latent = req
    else:
      self.cur_latent = self.sample_latent_from_surgical_info_dict(0)

  def get_current_latent(self):
    return self.cur_latent

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    # if agent holds the tool that he/she thinks as required,
    # then he/she asks the surgeon about the requirement
    nur_dir, nur_pos, nur_tool, _, nur_ask, _, _ = self.assumed_tup_states
    if (self.cur_latent == tho.Requirement.Nurse_Assist
        or nur_tool == self.cur_latent):
      if (tho.can_exchange_tool(
          nur_pos, nur_dir, self.policy_model.mdp.surgeon_pos) and not nur_ask):
        if np.random.random() < 0.3:
          return (tho.NurseAction.Ask_Requirement, None)

    oidx = self.policy_model.mdp.conv_sim_states_to_mdp_sidx(
        self.assumed_tup_states[:5])
    xidx = self.policy_model.conv_latent_to_idx(self.cur_latent)
    tup_aidx = self.policy_model.get_action(oidx, xidx)

    return self.policy_model.conv_idx_to_action(tup_aidx)[0]

  def set_latent(self, latent):
    self.cur_latent = latent

  def set_action(self, action):
    self.manual_action = action

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    prev_tup_states = self.assumed_tup_states
    obs_action, obs_state = self.observed_action_state(tup_actions,
                                                       tup_nxt_state)
    self.assumed_tup_states = obs_state

    nur_dir, nur_pos, nur_tool, sur_tool, nur_ask, sur_step, req = obs_state
    nur_act, sur_act = obs_action

    if req is not None:
      self.cur_latent = req
    else:
      # change if the surgeon has the currently-assumed tool
      if sur_tool == self.cur_latent:
        if not self.experienced:
          np_prop = np.ones(len(tho.Requirement))
          np_prop[self.cur_latent.value] = 0
          np_prop = np_prop / np_prop.sum()
          self.cur_latent = np.random.choice(list(tho.Requirement), p=np_prop)
        else:
          n_steps = len(self.surgical_steps)
          self.cur_latent = self.sample_latent_from_surgical_info_dict(
              (sur_step + 1) % n_steps)
      # not change
      else:
        pass

    return self.cur_latent
