import os
from typing import Hashable, Tuple
from stand_alone.app import AppInterface
from aic_domain.tool_handover_v2.simulator import ToolHandoverV2Simulator
from aic_domain.tool_handover_v2.mdp import MDP_ToolHandover_V2
import aic_domain.tool_handover_v2.define as tho
from aic_domain.tool_handover_v2.surgery_info import CABG_INFO
from aic_domain.tool_handover_v2.agent import (SurgeonAgent, PerfusionAgent,
                                               AnesthesiaAgent, NurseAgent)
from aic_domain.tool_handover_v2.nurse_mdp import THONursePolicy, MDP_THO_Nurse
from aic_domain.agent import InteractiveAgent


class ToolHandoverV2App(AppInterface):

  def __init__(self) -> None:
    super().__init__()
    self.image_dir = os.path.join(os.path.dirname(__file__), "../images/")

  def _init_game(self):
    self.width = CABG_INFO["width"]
    self.height = CABG_INFO["height"]

    TEMPERATURE = 0.3
    # nurse_agent = InteractiveAgent()
    nurse_mdp = MDP_THO_Nurse(**CABG_INFO)
    nurse_policy = THONursePolicy(nurse_mdp, TEMPERATURE)

    # Setting this variable False will emulate a novice nurse
    EXPERIENCED_NURSE = True
    self.nurse_agent = NurseAgent(nurse_policy, EXPERIENCED_NURSE)

    surgeon_agent = SurgeonAgent(CABG_INFO["surgeon_pos"])
    anes_agent = AnesthesiaAgent()
    perf_agent = PerfusionAgent()

    mdp = MDP_ToolHandover_V2(**CABG_INFO)
    self.game = ToolHandoverV2Simulator()
    self.game.init_game(mdp)
    self.game.set_autonomous_agent(nurse_agent=self.nurse_agent,
                                   surgeon_agent=surgeon_agent,
                                   anes_agent=anes_agent,
                                   perf_agent=perf_agent)

  def _init_gui(self):
    self.main_window.title("Tool Handover")
    self.canvas_width = 600
    self.canvas_height = 600

    return super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:

    agent_id = None
    action = None

    # nurse action
    if key_sym == "a":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.Rotate_Left, None)
    elif key_sym == "s":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.Rotate_180, None)
    elif key_sym == "d":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.Rotate_Right, None)
    elif key_sym == "w":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.Move_Forward, None)
    elif key_sym == "r":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.Ask_Requirement, None)
    elif key_sym == "q":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.Assist, None)
    elif key_sym == "z":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant1)
    elif key_sym == "x":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant2)
    elif key_sym == "c":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant3)
    elif key_sym == "v":
      agent_id = ToolHandoverV2Simulator.Nurse
      action = (tho.NurseAction.PickUp_Drop, tho.PickupLocation.Quadrant4)

    return (agent_id, tho.EventType.Action, action)

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    return super()._conv_mouse_to_agent_event(is_left, cursor_pos)

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()

  def _update_canvas_scene(self):
    data = self.game.get_env_info()
    if len(self.game.history) > 0:
      print(self.game.history[-1][2].name, self.game.history[-1][1][:2])

    x_unit = int(self.canvas_width / self.width)
    y_unit = int(self.canvas_height / self.height)

    self.clear_canvas()
    surgeon_pos = data["surgeon_pos"]
    self.create_circle((surgeon_pos[0] + 0.5) * x_unit,
                       (surgeon_pos[1] + 0.5) * y_unit, x_unit / 2, "yellow")
    perf_pos = data["perf_pos"]
    self.create_circle((perf_pos[0] + 0.5) * x_unit,
                       (perf_pos[1] + 0.5) * y_unit, x_unit / 2, "gray")
    anes_pos = data["anes_pos"]
    self.create_circle((anes_pos[0] + 0.5) * x_unit,
                       (anes_pos[1] + 0.5) * y_unit, x_unit / 2, "gray")

    patient_pos_size = data["patient_pos_size"]
    self.create_rectangle(patient_pos_size[0] * x_unit,
                          patient_pos_size[1] * y_unit,
                          (patient_pos_size[0] + patient_pos_size[2]) * x_unit,
                          (patient_pos_size[1] + patient_pos_size[3]) * y_unit,
                          "blue")
    table_blocks = data["table_blocks"]
    for table in table_blocks:
      self.create_rectangle(table[0] * x_unit, table[1] * y_unit,
                            (table[0] + 1) * x_unit, (table[1] + 1) * y_unit,
                            "green")

    vital_pos = data["vital_pos"]
    pat_vital = data["patient_vital"]
    self.create_text((vital_pos[0] + 0.5) * x_unit,
                     (vital_pos[1] + 0.5) * y_unit, pat_vital.name)

    nurse_pos = data["nurse_pos"]
    nurse_dir = data["nurse_dir"]
    self.create_circle((nurse_pos[0] + 0.5) * x_unit,
                       (nurse_pos[1] + 0.5) * y_unit, x_unit / 2, "red")
    if nurse_dir == tho.NurseDirection.Up:
      self.create_rectangle(
          (nurse_pos[0] + 0.4) * x_unit, (nurse_pos[1]) * y_unit,
          (nurse_pos[0] + 0.6) * x_unit, (nurse_pos[1] + 0.2) * y_unit, "red")
    elif nurse_dir == tho.NurseDirection.Left:
      self.create_rectangle(
          (nurse_pos[0]) * x_unit, (nurse_pos[1] + 0.4) * y_unit,
          (nurse_pos[0] + 0.2) * x_unit, (nurse_pos[1] + 0.6) * y_unit, "red")
    elif nurse_dir == tho.NurseDirection.Down:
      self.create_rectangle(
          (nurse_pos[0] + 0.4) * x_unit, (nurse_pos[1] + 0.8) * y_unit,
          (nurse_pos[0] + 0.6) * x_unit, (nurse_pos[1] + 1) * y_unit, "red")
    elif nurse_dir == tho.NurseDirection.Right:
      self.create_rectangle(
          (nurse_pos[0] + 0.8) * x_unit, (nurse_pos[1] + 0.4) * y_unit,
          (nurse_pos[0] + 1) * x_unit, (nurse_pos[1] + 0.6) * y_unit, "red")

    if data["surgeon_ready"]:
      self.create_text((surgeon_pos[0] + 0.5) * x_unit,
                       (surgeon_pos[1] + 0.1) * y_unit, "Ready")
    if data["anes_ready"]:
      self.create_text((anes_pos[0] + 0.5) * x_unit,
                       (anes_pos[1] + 0.1) * y_unit, "Ready")
    if data["perf_ready"]:
      self.create_text((perf_pos[0] + 0.5) * x_unit,
                       (perf_pos[1] + 0.1) * y_unit, "Ready")

    cur_step = data["cur_step"]
    self.create_text(self.canvas_width - 150, 20, f"Surgical Step: {cur_step}")
    cur_req = data["cur_requirement"]
    self.create_text(self.canvas_width - 150, 40, f"Requirement: {cur_req}")
    n_asked = data["nurse_asked"]
    self.create_text(self.canvas_width - 150, 60, f"Nurse Asked: {n_asked}")

    nurse_tool = data["nurse_tool"]
    surgeon_tool = data["surgeon_tool"]
    for tool, tb_zone in data["tool_table_zone"].items():
      if tool == nurse_tool:
        self.create_text((nurse_pos[0] + 0.5) * x_unit,
                         (nurse_pos[1] + 0.5) * y_unit, tool.name)
      elif tool == surgeon_tool:
        self.create_text((surgeon_pos[0] + 0.5) * x_unit,
                         (surgeon_pos[1] + 0.5) * y_unit, tool.name)
      else:
        table_lt = table_blocks[tb_zone[0]]
        if tb_zone[1] == 1:
          tool_pos = (table_lt[0] + 0.75, table_lt[1] + 0.35)
        elif tb_zone[1] == 2:
          tool_pos = (table_lt[0] + 0.25, table_lt[1] + 0.15)
        elif tb_zone[1] == 3:
          tool_pos = (table_lt[0] + 0.25, table_lt[1] + 0.65)
        elif tb_zone[1] == 4:
          tool_pos = (table_lt[0] + 0.75, table_lt[1] + 0.85)
        else:
          raise ValueError("Invalid tool zone")

        self.create_text(tool_pos[0] * x_unit, tool_pos[1] * y_unit, tool.name)


if __name__ == "__main__":
  app = ToolHandoverV2App()
  app.run()
