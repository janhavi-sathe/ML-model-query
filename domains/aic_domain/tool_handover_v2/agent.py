from aic_domain.agent import SimulatorAgent
import aic_domain.tool_handover_v2.define as tho
import random


class NoMindAgent(SimulatorAgent):

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


class AnesthesiaAgent(NoMindAgent):

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


class PerfusionAgent(NoMindAgent):

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


class SurgeonAgent(NoMindAgent):

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
