import os
from typing import Hashable, Tuple
from stand_alone.app import AppInterface
from .simulator import ToolHandoverSimulator
import aicoach.domains.tool_handover.define as tho


class ToolHandoverApp(AppInterface):

  def __init__(self) -> None:
    super().__init__()
    self.image_dir = os.path.join(os.path.dirname(__file__), "images/")

  def _init_game(self):
    self.game = ToolHandoverSimulator()
    self.game.init_game()

  def _init_gui(self):
    self.main_window.title("Tool Handover")
    self.canvas_width = 300
    self.canvas_height = 200

    icon_sz = 80
    self.nurse_position = (150, 60, icon_sz, icon_sz)
    self.surgeon_position = (220, 60, icon_sz, icon_sz)
    self.patient_position = (190, 120, 200, 80)
    self.table_position = (60, 110, 120, 120)
    self.label_text = (270, 20)
    self.label_text2 = (270, 35)

    self.tool_sz = 30
    self.table_part1 = (self.table_position[0] - 20,
                        self.table_position[1] - 35, self.tool_sz, self.tool_sz)
    self.table_part2 = (self.table_position[0] + 20,
                        self.table_position[1] - 35, self.tool_sz, self.tool_sz)
    self.table_part3 = (self.table_position[0] - 20,
                        self.table_position[1] - 10, self.tool_sz, self.tool_sz)
    self.table_part4 = (self.table_position[0] + 20,
                        self.table_position[1] - 10, self.tool_sz, self.tool_sz)
    return super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:

    agent_id = None
    action = None
    value = None

    # nurse action
    if key_sym == "r":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.Drop, None)
    elif key_sym == "f":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.PickUp, None)
    elif key_sym == "q":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.Move_hand, tho.Tool_Location.Table_1)
    elif key_sym == "w":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.Move_hand, tho.Tool_Location.Table_2)
    elif key_sym == "a":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.Move_hand, tho.Tool_Location.Table_3)
    elif key_sym == "s":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.Move_hand, tho.Tool_Location.Table_4)
    elif key_sym == "d":
      agent_id = ToolHandoverSimulator.Nurse
      action = (tho.NurseAction.Move_hand, tho.Tool_Location.Surgeon)

    # surgeon action
    if key_sym == "j":
      agent_id = ToolHandoverSimulator.Surgeon
      action = (tho.SurgeonAction.Change_View, None)
    elif key_sym == "h":
      agent_id = ToolHandoverSimulator.Surgeon
      action = (tho.SurgeonAction.Handover, None)
    elif key_sym == "n":
      agent_id = ToolHandoverSimulator.Surgeon
      action = (tho.SurgeonAction.Next_Step, None)

    return (agent_id, action, value)

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

    tool_types = data["tool_types"]
    patient_state = data["patient_state"]
    surgeon_sight = data["surgeon_sight"]
    surgical_step = data["surgical_step"]
    nurse_hand = data["nurse_hand"]
    tool_fornow = data["tool_fornow"]
    tool_locations = data["tool_locations"]
    current_step = data["current_step"]

    self.clear_canvas()

    # background
    self.create_image(*self.nurse_position, "nurse",
                      self.image_dir + "nurse.png")
    self.create_image(*self.surgeon_position, "surgeon",
                      self.image_dir + "surgeon.png")
    self.create_image(*self.table_position, "table",
                      self.image_dir + "table.png")
    self.create_image(*self.patient_position, "patient",
                      self.image_dir + "patient.png")

    # tools
    for idx, ttype in enumerate(tool_types):
      if ttype == tho.Tool_Type.Tool_1:
        image_file = "scalpel.png"
      elif ttype == tho.Tool_Type.Tool_2:
        image_file = "suture.png"
      elif ttype == tho.Tool_Type.Tool_3:
        image_file = "forceps.png"
      elif ttype == tho.Tool_Type.Tool_4:
        image_file = "scissors.png"

      if tool_locations[idx] == tho.Tool_Location.Nurse:
        img_pos = (self.nurse_position[0] + 20, self.nurse_position[1] - 20,
                   self.tool_sz, self.tool_sz)
      elif tool_locations[idx] == tho.Tool_Location.Surgeon:
        img_pos = (self.surgeon_position[0] + 20, self.surgeon_position[1] - 20,
                   self.tool_sz, self.tool_sz)
      elif tool_locations[idx] == tho.Tool_Location.Table_1:
        img_pos = self.table_part1
      elif tool_locations[idx] == tho.Tool_Location.Table_2:
        img_pos = self.table_part2
      elif tool_locations[idx] == tho.Tool_Location.Table_3:
        img_pos = self.table_part3
      elif tool_locations[idx] == tho.Tool_Location.Table_4:
        img_pos = self.table_part4

      self.create_image(*img_pos, ttype.name, self.image_dir + image_file)

    # surgical step
    self.create_text(*self.label_text, surgical_step.name)

    # patient state
    self.create_text(self.patient_position[0], self.patient_position[1],
                     patient_state.name)

    # surgeon sight
    if surgeon_sight == tho.SurgeonSight.Patient:
      self.create_line(self.surgeon_position[0] - 5,
                       self.surgeon_position[1] - 10, self.patient_position[0],
                       self.patient_position[1], 'black', 1)
      self.create_line(
          self.surgeon_position[0] + 5, self.surgeon_position[1] - 10,
          self.patient_position[0] + 0.3 * self.patient_position[2],
          self.patient_position[1], 'black', 1)
    else:
      self.create_line(self.surgeon_position[0] - 5,
                       self.surgeon_position[1] - 15,
                       self.table_position[0] + 0.3 * self.table_position[2],
                       self.table_position[1] - 0.4 * self.table_position[3],
                       'black', 1)
      self.create_line(self.surgeon_position[0] - 5,
                       self.surgeon_position[1] - 10,
                       self.table_position[0] + 0.3 * self.table_position[2],
                       self.table_position[1], 'black', 1)

    # nurse hand
    arm_st = (self.nurse_position[0] - 25, self.nurse_position[1] + 10)
    hand_sz = 5
    if nurse_hand == tho.Tool_Location.Surgeon:
      hand_coord = (self.surgeon_position[0] - 30, self.surgeon_position[1])
    elif nurse_hand == tho.Tool_Location.Table_1:
      hand_coord = (self.table_part1[0] + 15, self.table_part1[1] - 5)
    elif nurse_hand == tho.Tool_Location.Table_2:
      hand_coord = (self.table_part2[0] + 15, self.table_part2[1] - 5)
    elif nurse_hand == tho.Tool_Location.Table_3:
      hand_coord = (self.table_part3[0] + 15, self.table_part3[1] - 5)
    elif nurse_hand == tho.Tool_Location.Table_4:
      hand_coord = (self.table_part4[0] + 15, self.table_part4[1] - 5)

    mid_coord = (int(0.5 * (hand_coord[0] + arm_st[0])),
                 int(0.5 * (hand_coord[1] + arm_st[1])) + 10)

    self.create_line(*arm_st, *mid_coord, "red", 3)
    self.create_line(*mid_coord, *hand_coord, "red", 3)
    self.create_circle(*hand_coord, hand_sz, "red")

    # tool for now
    text_tool = "None"
    if tool_fornow is not None:
      text_tool = tool_fornow.name
    self.create_text(*self.label_text2, text_tool)


if __name__ == "__main__":
  app = ToolHandoverApp()
  app.run()
