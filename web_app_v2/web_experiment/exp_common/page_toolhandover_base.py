from typing import Mapping, Any, Sequence, List
from web_experiment.exp_common.helper_surgery import (
    toolhandover_game_scene, toolhandover_game_scene_names)
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base_surgery import (SurgeryPageBase,
                                                         SurgeryUserData)
from aic_domain.tool_handover_v2.simulator import ToolHandoverV2Simulator
import aic_domain.tool_handover_v2.define as tho
from aic_domain.tool_handover_v2.surgery_info import CABG_INFO
from aic_domain.tool_handover_v2.agent import (SurgeonAgent, PerfusionAgent,
                                               AnesthesiaAgent)


class ToolHandoverGamePageBase(SurgeryPageBase):
  N_STAY = tho.NurseAction.Stay.name
  N_ROTATE_L = tho.NurseAction.Rotate_Left.name
  N_ROTATE_180 = tho.NurseAction.Rotate_180.name
  N_ROTATE_R = tho.NurseAction.Rotate_Right.name
  N_ASK_REQUIREMENT = tho.NurseAction.Ask_Requirement.name
  N_ASSIST = tho.NurseAction.Assist.name
  N_PICKUPDROP = tho.NurseAction.PickUp_Drop.name

  ACTION_BUTTONS = [
      N_STAY, N_ROTATE_L, N_ROTATE_180, N_ROTATE_R, N_ASK_REQUIREMENT, N_ASSIST,
      N_PICKUPDROP
  ]

  CONTROL_PANEL = "control_panel"

  def __init__(self, manual_latent_selection, surgery_info) -> None:
    super().__init__(True, True, True, EDomainType.ToolHandover)
    self._MANUAL_SELECTION = manual_latent_selection
    self._SURGERY_INFO = surgery_info

    self._N = ToolHandoverV2Simulator.Nurse
    self._S = ToolHandoverV2Simulator.Surgeon
    self._P = ToolHandoverV2Simulator.Anes
    self._A = ToolHandoverV2Simulator.Perf

  def init_user_data(self, user_game_data: SurgeryUserData):
    user_game_data.data[SurgeryUserData.GAME_DONE] = False

    game = user_game_data.get_game_ref()
    if game is None:
      game = ToolHandoverV2Simulator()

      user_game_data.set_game(game)
    game.init_game(**CABG_INFO)
    surgeon_agent = SurgeonAgent(CABG_INFO["surgeon_pos"])
    perf_agent = PerfusionAgent()
    anes_agent = AnesthesiaAgent()
    game.set_autonomous_agent(surgeon_agent=surgeon_agent,
                              perf_agent=perf_agent,
                              anes_agent=anes_agent)

  def get_updated_drawing_info(self,
                               user_data: SurgeryUserData,
                               clicked_button: str = None,
                               dict_prev_scene_data: Mapping[str, Any] = None):
    if dict_prev_scene_data is None:
      drawing_objs = self._get_init_drawing_objects(user_data)
      commands = self._get_init_commands(user_data)
      animations = None
    else:
      drawing_objs = self._get_updated_drawing_objects(user_data,
                                                       dict_prev_scene_data)
      commands = self._get_button_commands(clicked_button, user_data)
      game = user_data.get_game_ref()
      animations = None
      if clicked_button in self.ACTION_BUTTONS:
        animations = self._get_animations(dict_prev_scene_data,
                                          game.get_env_info())
    drawing_order = self._get_drawing_order(user_data)

    return commands, drawing_objs, drawing_order, animations

  def button_clicked(self, user_game_data: SurgeryUserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if self.is_sel_latent_btn(clicked_btn):
      latent = self.selbtn2latent(clicked_btn)
      if latent is not None:
        _, _, _, done = self.action_event(user_game_data, latent)
        if done:
          self._on_game_finished(user_game_data)

        return
    elif clicked_btn == self.N_PICKUPDROP:
      user_game_data.data[SurgeryUserData.SELECT_MH] = True
      return
    elif clicked_btn in self.ACTION_BUTTONS:  # other action buttons
      _, _, _, done = self.action_event(user_game_data, clicked_btn)
      if done:
        self._on_game_finished(user_game_data)
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, user_game_data: SurgeryUserData):
    return ("Control nurse with the buttons.")

  def _get_drawing_order(self, user_game_data: SurgeryUserData):
    dict_game = user_game_data.get_game_ref().get_env_info()
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_game_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_game_data))
    drawing_order = drawing_order + self.ACTION_BUTTONS

    drawing_order.append(self.TEXT_INSTRUCTION)

    drawing_order.append(self.CONTROL_PANEL)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_game_data: SurgeryUserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    dict_game = user_game_data.get_game_ref().get_env_info()

    game_objs = self._game_scene(dict_game,
                                 user_game_data,
                                 include_background=True)
    for obj in game_objs:
      dict_objs[obj.name] = obj

    overlay_objs = self._game_overlay(dict_game, user_game_data)
    for obj in overlay_objs:
      dict_objs[obj.name] = obj

    disable_status = self._get_action_btn_disable_state(user_game_data,
                                                        dict_game)
    objs = self._get_btn_actions(dict_game, user_game_data, *disable_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_action_btn_disable_state(self, user_data: SurgeryUserData,
                                    game_env: Mapping[Any, Any]):
    game = user_data.get_game_ref()
    game_env = game.get_env_info()

    game_done = user_data.data[SurgeryUserData.GAME_DONE]

    if game_done:
      return True, True, True, True, True, True, True

    return False, False, False, False, False, False, False

  def _get_btn_actions(
      self,
      game_env: Mapping[Any, Any],
      user_game_data: SurgeryUserData,
      disable_stay: bool = False,
      disable_rotate_l: bool = False,
      disable_rotate_180: bool = False,
      disable_rotate_r: bool = False,
      disable_ask: bool = False,
      disable_assist: bool = False,
      disable_pickdrop: bool = False) -> Sequence[co.DrawingObject]:

    game_width = self.GAME_WIDTH
    game_right = self.GAME_RIGHT
    ctrl_btn_w = int(game_width / 12)
    ctrl_btn_w_half = int(game_width / 24)
    x_ctrl_cen = int(game_right + (co.CANVAS_WIDTH - game_right) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    x_joy_cen = int(x_ctrl_cen - ctrl_btn_w * 1.5)

    font_size = 20
    btn_stay = co.ButtonRect(self.N_STAY, (x_ctrl_cen - int(ctrl_btn_w * 1.5),
                                           y_ctrl_cen - int(ctrl_btn_w * 1.8)),
                             (ctrl_btn_w * 3, ctrl_btn_w),
                             font_size,
                             "Stay",
                             disable=disable_stay)
    btn_rot_l = co.ButtonRect(self.N_ROTATE_L,
                              (x_ctrl_cen - int(ctrl_btn_w * 1.5),
                               y_ctrl_cen - int(ctrl_btn_w * 0.6)),
                              (ctrl_btn_w * 3, ctrl_btn_w),
                              font_size,
                              "Rotate Left",
                              disable=disable_rotate_l)
    btn_rot_180 = co.ButtonRect(self.N_ROTATE_180,
                                (x_ctrl_cen - int(ctrl_btn_w * 1.5),
                                 y_ctrl_cen + int(ctrl_btn_w * 0.6)),
                                (ctrl_btn_w * 3, ctrl_btn_w),
                                font_size,
                                "Rotate 180",
                                disable=disable_rotate_180)
    btn_rot_r = co.ButtonRect(self.N_ROTATE_R,
                              (x_ctrl_cen - int(ctrl_btn_w * 1.5),
                               y_ctrl_cen + int(ctrl_btn_w * 1.8)),
                              (ctrl_btn_w * 3, ctrl_btn_w),
                              font_size,
                              "Rotate Right",
                              disable=disable_rotate_r)
    btn_ask = co.ButtonRect(self.N_ASK_REQUIREMENT,
                            (x_ctrl_cen - int(ctrl_btn_w * 1.5),
                             y_ctrl_cen + int(ctrl_btn_w * 3.0)),
                            (ctrl_btn_w * 3, ctrl_btn_w),
                            font_size,
                            "Ask Requirement",
                            disable=disable_ask)

    btn_assist = co.ButtonRect(self.N_ASSIST,
                               (x_ctrl_cen + int(ctrl_btn_w * 1.5),
                                y_ctrl_cen - int(ctrl_btn_w * 1.8)),
                               (ctrl_btn_w * 3, ctrl_btn_w),
                               font_size,
                               "Assist",
                               disable=disable_assist)

    pickdrop_label = "Drop"
    if game_env["nurse_tool"] == tho.Requirement.Hand_Only:
      pickdrop_label = "Pick Up"

    btn_pickdrop = co.ButtonRect(self.N_PICKUPDROP,
                                 (x_ctrl_cen + int(ctrl_btn_w * 1.5),
                                  y_ctrl_cen - int(ctrl_btn_w * 0.6)),
                                 (ctrl_btn_w * 3, ctrl_btn_w),
                                 font_size,
                                 pickdrop_label,
                                 disable=disable_pickdrop)

    list_buttons = [
        btn_stay, btn_rot_l, btn_rot_180, btn_rot_r, btn_ask, btn_assist,
        btn_pickdrop
    ]
    return list_buttons

  def _get_control_panel(self):
    game_right = self.GAME_RIGHT

    frame = co.Rectangle(self.CONTROL_PANEL,
                         (game_right, int(co.CANVAS_HEIGHT * 0.15)), (350, 350),
                         "black", "black")
    return [frame]

  def _on_action_taken(self, user_game_data: SurgeryUserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    '''
    user_cur_game_data: NOTE - values will be updated
    '''

    pass

  def _on_game_finished(self, user_game_data: SurgeryUserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.data[SurgeryUserData.GAME_DONE] = True

    # update score
    game = user_game_data.get_game_ref()
    user_game_data.data[SurgeryUserData.SCORE] = game.current_step

  def _get_updated_drawing_objects(
      self,
      user_data: SurgeryUserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    dict_game = user_data.get_game_ref().get_env_info()
    dict_objs = {}
    game_updated = (dict_prev_game["current_step"] != dict_game["current_step"])
    if game_updated:
      for obj in self._game_scene(dict_game, user_data, False):
        dict_objs[obj.name] = obj

      obj = self._get_score_obj(user_data)
      dict_objs[obj.name] = obj

    for obj in self._game_overlay(dict_game, user_data):
      dict_objs[obj.name] = obj

    for obj in self._get_instruction_objs(user_data):
      dict_objs[obj.name] = obj

    for obj in self._get_control_panel():
      dict_objs[obj.name] = obj

    disable_status = self._get_action_btn_disable_state(user_data, dict_game)
    objs = self._get_btn_actions(dict_game, user_data, *disable_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    return None

  def _get_animations(self, dict_prev_game: Mapping[str, Any],
                      dict_cur_game: Mapping[str, Any]):

    return []

  def action_event(self, user_game_data: SurgeryUserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    '''
    game = user_game_data.get_game_ref()

    action = None
    if clicked_btn == self.N_STAY:
      action = (tho.NurseAction.Stay, None)
    elif clicked_btn == self.N_ROTATE_R:
      action = (tho.NurseAction.Rotate_Right, None)
    elif clicked_btn == self.N_ROTATE_L:
      action = (tho.NurseAction.Rotate_Left, None)
    elif clicked_btn == self.N_ROTATE_180:
      action = (tho.NurseAction.Rotate_180, None)
    elif clicked_btn == self.N_ASSIST:
      action = (tho.NurseAction.Assist, None)
    elif clicked_btn == self.N_ASK_REQUIREMENT:
      action = (tho.NurseAction.Ask_Requirement, None)
    elif "mh" in clicked_btn:
      loc_id = int(clicked_btn.split("mh")[1])
      action = (tho.NurseAction.PickUp_Drop, tho.PickupLocation(loc_id))

    # should not happen
    assert action is not None
    assert not game.is_finished()

    game.event_input(self._N, tho.EventType.Action, action)

    # take actions
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    return ([], [], [], game.is_finished())

  def _game_overlay(self, game_env,
                    user_data: SurgeryUserData) -> List[co.DrawingObject]:

    grid_w = game_env["width"]
    grid_h = game_env["height"]

    def coord_2_canvas(coord_x, coord_y):
      x = int(self.GAME_LEFT + coord_x / grid_w * self.GAME_WIDTH)
      y = int(self.GAME_TOP + coord_y / grid_h * self.GAME_HEIGHT)
      return (x, y)

    def size_2_canvas(width, height):
      w = int(width / grid_w * self.GAME_WIDTH)
      h = int(height / grid_h * self.GAME_HEIGHT)
      return (w, h)

    overlay_obs = []

    if user_data.data[SurgeryUserData.SELECT_MH]:
      obj = co.Rectangle(co.SEL_LAYER, (self.GAME_LEFT, self.GAME_TOP),
                         (self.GAME_WIDTH, self.GAME_HEIGHT),
                         fill_color="white",
                         alpha=0.8)
      overlay_obs.append(obj)

      radius = size_2_canvas(15, 0)[0]
      font_size = 20

      nurse_pos = game_env["nurse_pos"]
      nurse_dir = game_env["nurse_dir"]
      target_pos = tho.get_target_pos(nurse_pos, nurse_dir)

      for idx, loc in enumerate(tho.PickupLocation):
        if loc == tho.PickupLocation.Quadrant1:
          img_pos = (target_pos[0] + 0.75, target_pos[1] + 0.25)
        elif loc == tho.PickupLocation.Quadrant2:
          img_pos = (target_pos[0] + 0.25, target_pos[1] + 0.25)
        elif loc == tho.PickupLocation.Quadrant3:
          img_pos = (target_pos[0] + 0.25, target_pos[1] + 0.75)
        elif loc == tho.PickupLocation.Quadrant4:
          img_pos = (target_pos[0] + 0.75, target_pos[1] + 0.75)
        else:
          raise ValueError("Unknown location")

        # store latent buttons in order of tool type
        obj = co.SelectingCircle("latentmh" + str(idx),
                                 coord_2_canvas(img_pos[0], img_pos[1]), radius,
                                 font_size, "")
        overlay_obs.append(obj)

    return overlay_obs

  def _game_overlay_names(self, game_env, user_data: SurgeryUserData) -> List:
    overlay_names = []
    if user_data.data[SurgeryUserData.SELECT_MH]:
      overlay_names.append(co.SEL_LAYER)

      for idx, loc in enumerate(tho.PickupLocation):
        overlay_names.append("latentmh" + str(idx))

    return overlay_names

  def _game_scene(self,
                  game_env,
                  user_data: SurgeryUserData,
                  include_background: bool = True) -> List[co.DrawingObject]:

    game_ltwh = (self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH,
                 self.GAME_HEIGHT)
    return toolhandover_game_scene(game_env, game_ltwh, include_background)

  def _game_scene_names(self, game_env, user_data: SurgeryUserData) -> List:

    def is_visible(img_name):
      return True

    return toolhandover_game_scene_names(game_env, is_visible)

  def latent2selbtn(self, latent):
    return "latent" + str(latent)

  def selbtn2latent(self, sel_btn_name):
    if sel_btn_name[:6] == "latent":
      return sel_btn_name[6:]

    return None

  def is_sel_latent_btn(self, sel_btn_name):
    return sel_btn_name[:6] == "latent"
