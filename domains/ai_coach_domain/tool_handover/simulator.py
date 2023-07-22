from typing import Hashable, Mapping
import numpy as np
from ai_coach_domain.simulator import Simulator
from ai_coach_domain.tool_handover.mdp import MDP_ToolHandover
import ai_coach_domain.tool_handover.define as tho


class ToolHandoverSimulator(Simulator):
  Surgeon = 0
  Nurse = 1

  def __init__(self) -> None:
    super().__init__(0)

    self.list_tool_types = list(tho.Tool_Type)
    self.mmdp = MDP_ToolHandover(self.list_tool_types)
    self.max_steps = 50

  def init_game(self):
    self.reset_game()

  def reset_game(self):
    self.current_step = 0
    self.history = []

    self.patient_state = tho.PatientState.Stable
    self.surgeon_sight = tho.SurgeonSight.Patient
    self.surgical_step = tho.SurgicalStep.Step_0
    self.nurse_hand = tho.Tool_Location.Surgeon

    self.tool_locations = [
        tho.Tool_Location.Table_1, tho.Tool_Location.Table_2,
        tho.Tool_Location.Table_3, tho.Tool_Location.Table_4
    ]
    self.surgeon_action = None
    self.nurse_action = None

  def get_num_agents(self):
    return 2

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    tup_actions = tuple(
        [map_agent_2_action[idx] for idx in range(self.get_num_agents())])
    action_idx = self.mmdp.conv_sim_actions_to_mdp_aidx(tup_actions)

    tup_state = (self.patient_state, self.surgeon_sight, self.surgical_step,
                 self.nurse_hand, self.tool_locations)
    state_idx = self.mmdp.conv_sim_states_to_mdp_sidx(tup_state)

    self.history.append((tup_state, tup_actions, None))

    next_state_idx = self.mmdp.transition(state_idx, action_idx)
    (self.patient_state, self.surgeon_sight, self.surgical_step,
     self.nurse_hand, self.tool_locations
     ) = self.mmdp.conv_mdp_sidx_to_sim_states(next_state_idx)
    self.current_step += 1

  def event_input(self, agent: Hashable, event_type: Hashable, value=None):
    if agent == self.Surgeon:
      if event_type[0] in tho.SurgeonAction:
        self.surgeon_action = event_type
    elif agent == self.Nurse:
      if event_type[0] in tho.NurseAction:
        self.nurse_action = event_type

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:
    dict_agent_action = {}

    if self.surgeon_action is not None:
      dict_agent_action[self.Surgeon] = self.surgeon_action
    else:
      dict_agent_action[self.Surgeon] = (tho.SurgeonAction.Stay, None)

    if self.nurse_action is not None:
      dict_agent_action[self.Nurse] = self.nurse_action
    else:
      dict_agent_action[self.Nurse] = (tho.NurseAction.Stay, None)

    self.surgeon_action = None
    self.nurse_action = None

    return dict_agent_action

  def is_finished(self) -> bool:
    tup_state = (self.patient_state, self.surgeon_sight, self.surgical_step,
                 self.nurse_hand, self.tool_locations)

    state_idx = self.mmdp.conv_sim_states_to_mdp_sidx(tup_state)
    return self.mmdp.is_terminal(state_idx)

  def get_score(self):
    return -self.get_current_step()

  def get_env_info(self):
    env_info = {"tool_types": self.list_tool_types}
    env_info["patient_state"] = self.patient_state
    env_info["surgeon_sight"] = self.surgeon_sight
    env_info["surgical_step"] = self.surgical_step
    env_info["nurse_hand"] = self.nurse_hand
    env_info["tool_locations"] = self.tool_locations
    env_info["current_step"] = self.current_step

    return env_info
