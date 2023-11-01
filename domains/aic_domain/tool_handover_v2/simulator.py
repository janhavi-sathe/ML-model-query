from typing import Hashable, Mapping
import numpy as np
from aic_domain.simulator import Simulator
from aic_domain.tool_handover_v2.mdp import MDP_ToolHandover_V2
import aic_domain.tool_handover_v2.define as tho
from aic_domain.agent import SimulatorAgent, InteractiveAgent


class ToolHandoverV2Simulator(Simulator):
  Nurse = 0
  Surgeon = 1
  Anes = 2
  Perf = 3
  Score = 0

  def __init__(self) -> None:
    super().__init__(0)

    self.max_steps = 500
    self.nurse_agent = None
    self.surgeon_agent = None
    self.anes_agent = None
    self.perf_agent = None

  def init_game(self, mdp_tool_handover: MDP_ToolHandover_V2):
    self.mmdp = mdp_tool_handover
    self.reset_game()

  def set_autonomous_agent(
      self,
      nurse_agent: SimulatorAgent = InteractiveAgent(),
      surgeon_agent: SimulatorAgent = InteractiveAgent(),
      anes_agent: SimulatorAgent = InteractiveAgent(),
      perf_agent: SimulatorAgent = InteractiveAgent(),
  ):
    self.nurse_agent = nurse_agent
    self.surgeon_agent = surgeon_agent
    self.anes_agent = anes_agent
    self.perf_agent = perf_agent

    self.agents = [
        self.nurse_agent, self.surgeon_agent, self.anes_agent, self.perf_agent
    ]

    for idx, agent in enumerate(self.agents):
      agent.init_latent(self.get_state_for_each_agent(idx))

  def reset_game(self):
    self.current_step = 0
    self.history = []

    self.patient_vital = tho.PatientVital.Stable
    self.nurse_dir = self.mmdp.nurse_init_dir
    self.nurse_pos = self.mmdp.nurse_init_pos
    self.nurse_tool = tho.Requirement.Hand_Only
    self.surgeon_tool = tho.Requirement.Hand_Only
    self.surgeon_ready = False
    self.perf_ready = False
    self.anes_ready = False
    self.cur_step = 0
    self.cur_requirement = self.mmdp.surgical_steps[self.cur_step][
        self.patient_vital][0][0]
    self.nurse_asked = False

    if self.nurse_agent is not None:
      for idx, agent in enumerate(self.agents):
        agent.init_latent(self.get_state_for_each_agent(idx))

  def get_state_for_each_agent(self, agent_idx=-1):
    return (self.patient_vital, self.nurse_dir, self.nurse_pos, self.nurse_tool,
            self.surgeon_tool, self.surgeon_ready, self.perf_ready,
            self.anes_ready, self.cur_step, self.cur_requirement,
            self.nurse_asked)

  def get_num_agents(self):
    return 4

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    tup_actions = tuple(
        [map_agent_2_action[idx] for idx in range(self.get_num_agents())])
    action_idx = self.mmdp.conv_sim_actions_to_mdp_aidx(tup_actions)

    tup_state = self.get_state_for_each_agent()
    state_idx = self.mmdp.conv_sim_states_to_mdp_sidx(tup_state)

    self.history.append(
        (tup_state, tup_actions, self.nurse_agent.get_current_latent()))

    next_state_idx = self.mmdp.transition(state_idx, action_idx)
    (self.patient_vital, self.nurse_dir, self.nurse_pos, self.nurse_tool,
     self.surgeon_tool, self.surgeon_ready, self.perf_ready, self.anes_ready,
     self.cur_step, self.cur_requirement,
     self.nurse_asked) = self.mmdp.conv_mdp_sidx_to_sim_states(next_state_idx)
    self.current_step += 1

    # mental state
    for idx, agent in enumerate(self.agents):
      agent.update_mental_state(tup_state, tup_actions,
                                self.get_state_for_each_agent(idx))

  def event_input(self, agent: Hashable, event_type: Hashable, value=None):
    if agent == self.Nurse:
      if event_type == tho.EventType.Action:
        self.nurse_agent.set_action(value)
      elif event_type == tho.EventType.Set_Latent:
        self.nurse_agent.set_latent(value)
    elif agent == self.Surgeon:
      if event_type == tho.EventType.Action:
        self.surgeon_agent.set_action(value)
      elif event_type == tho.EventType.Set_Latent:
        self.surgeon_agent.set_latent(value)
    elif agent == self.Anes:
      if event_type == tho.EventType.Action:
        self.anes_agent.set_action(value)
      elif event_type == tho.EventType.Set_Latent:
        self.anes_agent.set_latent(value)
    elif agent == self.Perf:
      if event_type == tho.EventType.Action:
        self.perf_agent.set_action(value)
      elif event_type == tho.EventType.Set_Latent:
        self.perf_agent.set_latent(value)

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:
    dict_agent_action = {}
    nurse_action = self.nurse_agent.get_action(
        self.get_state_for_each_agent(self.Nurse))
    dict_agent_action[self.Nurse] = (nurse_action if nurse_action is not None
                                     else (tho.NurseAction.Stay, None))

    surgeon_action = self.surgeon_agent.get_action(
        self.get_state_for_each_agent(self.Surgeon))
    dict_agent_action[self.Surgeon] = (surgeon_action if surgeon_action
                                       is not None else tho.SurgeonAction.Stay)

    anes_action = self.anes_agent.get_action(
        self.get_state_for_each_agent(self.Anes))
    dict_agent_action[self.Anes] = (anes_action if anes_action is not None else
                                    tho.AnesthesiaAction.Stay)

    perf_action = self.perf_agent.get_action(
        self.get_state_for_each_agent(self.Perf))
    dict_agent_action[self.Perf] = (perf_action if perf_action is not None else
                                    tho.PerfusionAction.Stay)

    return dict_agent_action

  def is_finished(self) -> bool:
    if super().is_finished():
      return True

    tup_state = self.get_state_for_each_agent()

    state_idx = self.mmdp.conv_sim_states_to_mdp_sidx(tup_state)
    return self.mmdp.is_terminal(state_idx)

  def get_score(self):
    return -self.get_current_step()

  def get_env_info(self):
    env_info = {
        "width": self.mmdp.width,
        "height": self.mmdp.height,
        "surgeon_pos": self.mmdp.surgeon_pos,
        "patient_pos_size": self.mmdp.patient_pos_size,
        "perf_pos": self.mmdp.perf_pos,
        "anes_pos": self.mmdp.anes_pos,
        "nurse_init_pos": self.mmdp.nurse_init_pos,
        "nurse_init_dir": self.mmdp.nurse_init_dir,
        "nurse_possible_pos": self.mmdp.nurse_possible_pos,
        "table_blocks": self.mmdp.table_blocks,
        "vital_pos": self.mmdp.vital_pos,
        "surgical_steps": self.mmdp.surgical_steps,
        "tool_table_zone": self.mmdp.tool_table_zone,
        "patient_vital": self.patient_vital,
        "nurse_dir": self.nurse_dir,
        "nurse_pos": self.nurse_pos,
        "nurse_tool": self.nurse_tool,
        "surgeon_tool": self.surgeon_tool,
        "surgeon_ready": self.surgeon_ready,
        "perf_ready": self.perf_ready,
        "anes_ready": self.anes_ready,
        "cur_step": self.cur_step,
        "cur_requirement": self.cur_requirement,
        "nurse_asked": self.nurse_asked,
        "current_step": self.current_step
    }

    return env_info
