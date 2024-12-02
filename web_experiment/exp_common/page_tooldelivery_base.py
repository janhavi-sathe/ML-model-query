from typing import Mapping, Any, Sequence, List
from aicoach.domains.tooldelivery.tooldelivery_v3_state_action import (ActionCN,
                                                                       ActionSN,
                                                                       ActionAS,
                                                                       ToolLoc)
from aicoach.domains.tooldelivery.simulator import ToolDeliverySimulator
from web_experiment.exp_common.helper_handover import (
    tooldelivery_game_scene, tooldelivery_game_scene_names)
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import ExperimentPageBase, Exp1UserData


class ToolDeliveryGamePageBase(ExperimentPageBase):
  CN_STAY = ActionCN.STAY.name
  CN_UP = ActionCN.MOVE_UP.name
  CN_DOWN = ActionCN.MOVE_DOWN.name
  CN_LEFT = ActionCN.MOVE_LEFT.name
  CN_RIGHT = ActionCN.MOVE_RIGHT.name
  CN_PICKUP = ActionCN.PICKUP.name
  CN_HANDOVER = ActionCN.HANDOVER.name

  SN_STAY = ActionSN.STAY.name
  SN_HO_SCALPEL = ActionSN.HO_SCALPEL.name
  SN_HO_SUTURE = ActionSN.HO_SUTURE.name
  SN_ASKTOOL = ActionSN.ASKTOOL.name
  SN_SCALPEL_RELATED = ActionSN.SCALPEL_RELATED.name
  SN_SUTURE_RELATED = ActionSN.SUTURE_RELATED.name

  AS_STAY = ActionAS.STAY.name
  AS_HO_SCALPEL = ActionAS.HO_SCALPEL.name
  AS_USE_SCALPEL = ActionAS.USE_SCALPEL.name
  AS_USE_SUTURE = ActionAS.USE_SUTURE.name

  ACTION_BUTTONS = [
      CN_STAY,
      CN_UP,
      CN_DOWN,
      CN_LEFT,
      CN_RIGHT,
      CN_PICKUP,
      CN_HANDOVER,
      # SN_STAY, SN_HO_SCALPEL, SN_HO_SUTURE, SN_ASKTOOL, SN_SCALPEL_RELATED, SN_SUTURE_RELATED,
      # AS_STAY, AS_HO_SCALPEL, AS_USE_SCALPEL, AS_USE_SUTURE
  ]

  # ACTION_BUTTONS = ["next"]

  def __init__(self,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(True, True, True, EDomainType.ToolDelivery)
    self._MANUAL_SELECTION = manual_latent_selection
    self._GAME_MAP = game_map

    self._PROMPT_ON_CHANGE = prompt_on_change
    self._PROMPT_FREQ = prompt_freq
    self._AUTO_PROMPT = auto_prompt

    self._CN = ToolDeliverySimulator.CN
    self._SN = ToolDeliverySimulator.SN
    self._AS = ToolDeliverySimulator.AS

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False

    game = user_game_data.get_game_ref()
    if game is None:
      game = ToolDeliverySimulator()

      user_game_data.set_game(game)
    game.init_game()

  def get_updated_drawing_info(self,
                               user_data: Exp1UserData,
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

  def button_clicked(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    return: commands, drawing_objs, drawing_order, animations
      drawing info
    '''

    if clicked_btn in self.ACTION_BUTTONS:
      _, _, _, done = self.action_event(user_game_data, clicked_btn)
      if done:
        self._on_game_finished(user_game_data)
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return (
        "Control the CN with the buttons to progress through the simulation.")

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    dict_game = user_game_data.get_game_ref().get_env_info()
    drawing_order = []
    drawing_order.append(self.GAME_BORDER)

    drawing_order = (drawing_order +
                     self._game_scene_names(dict_game, user_game_data))
    drawing_order = (drawing_order +
                     self._game_overlay_names(dict_game, user_game_data))
    drawing_order = drawing_order + self.ACTION_BUTTONS

    drawing_order.append(self.TEXT_INSTRUCTION)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
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
    objs = self._get_btn_actions(dict_game, *disable_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_action_btn_disable_state(self, user_data: Exp1UserData,
                                    game_env: Mapping[Any, Any]):
    game = user_data.get_game_ref()
    game_env = game.get_env_info()

    game_done = user_data.data[Exp1UserData.GAME_DONE]

    if game_done:
      return True, True, True, True, True, True, True

    pickup_ok = True
    handover_ok = True
    cn_pos = game_env["CN_pos"]
    cabinet_loc = (game_env["Cabinet_pos_size"][0],
                   game_env["Cabinet_pos_size"][1])
    storage_loc = (game_env["Storage_pos_size"][0],
                   game_env["Storage_pos_size"][1])
    handover_loc = (game_env["Handover_pos_size"][0],
                    game_env["Handover_pos_size"][1])
    list_spare_tool_loc = [
        game_env["Scalpel_stored"], game_env["Suture_stored"]
    ]

    if cn_pos not in [cabinet_loc, storage_loc]:
      pickup_ok = False
    if cn_pos != handover_loc:
      handover_ok = False
    if cn_pos == cabinet_loc and ToolLoc.CABINET not in list_spare_tool_loc:
      pickup_ok = False
    if cn_pos == storage_loc and ToolLoc.STORAGE not in list_spare_tool_loc:
      pickup_ok = False
    if ToolLoc.CN not in list_spare_tool_loc:
      handover_ok = False

    print(cn_pos, cabinet_loc, storage_loc, handover_loc, list_spare_tool_loc,
          cn_pos in [cabinet_loc, storage_loc])
    print(ToolLoc.CABINET in list_spare_tool_loc, pickup_ok, handover_ok)
    return False, False, not pickup_ok, not handover_ok

  def _get_btn_actions(
      self,
      game_env: Mapping[Any, Any],
      disable_move: bool = False,
      disable_stay: bool = False,
      disable_pickup: bool = False,
      disable_handover: bool = False) -> Sequence[co.DrawingObject]:

    game_width = self.GAME_WIDTH
    game_right = self.GAME_RIGHT
    ctrl_btn_w = int(game_width / 12)
    ctrl_btn_w_half = int(game_width / 24)
    x_ctrl_cen = int(game_right + (co.CANVAS_WIDTH - game_right) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    x_joy_cen = int(x_ctrl_cen - ctrl_btn_w * 1.5)
    btn_stay = co.JoystickStay((x_joy_cen, y_ctrl_cen),
                               ctrl_btn_w,
                               disable=disable_stay,
                               name=self.CN_STAY)
    btn_up = co.JoystickUp((x_joy_cen, y_ctrl_cen - ctrl_btn_w_half),
                           ctrl_btn_w,
                           disable=disable_move,
                           name=self.CN_UP)
    btn_right = co.JoystickRight((x_joy_cen + ctrl_btn_w_half, y_ctrl_cen),
                                 ctrl_btn_w,
                                 disable=disable_move,
                                 name=self.CN_RIGHT)
    btn_down = co.JoystickDown((x_joy_cen, y_ctrl_cen + ctrl_btn_w_half),
                               ctrl_btn_w,
                               disable=disable_move,
                               name=self.CN_DOWN)
    btn_left = co.JoystickLeft((x_joy_cen - ctrl_btn_w_half, y_ctrl_cen),
                               ctrl_btn_w,
                               disable=disable_move,
                               name=self.CN_LEFT)

    font_size = 20
    btn_pickup = co.ButtonRect(self.CN_PICKUP,
                               (x_ctrl_cen + int(ctrl_btn_w * 1.5),
                                y_ctrl_cen - int(ctrl_btn_w * 0.6)),
                               (ctrl_btn_w * 2, ctrl_btn_w),
                               font_size,
                               "Pick Up",
                               disable=disable_pickup)
    btn_handover = co.ButtonRect(self.CN_HANDOVER,
                                 (x_ctrl_cen + int(ctrl_btn_w * 1.5),
                                  y_ctrl_cen + int(ctrl_btn_w * 0.6)),
                                 (ctrl_btn_w * 2, ctrl_btn_w),
                                 font_size,
                                 "Handover",
                                 disable=disable_handover)

    list_buttons = [
        btn_stay, btn_up, btn_right, btn_down, btn_left, btn_pickup,
        btn_handover
    ]
    return list_buttons

  def _on_action_taken(self, user_game_data: Exp1UserData,
                       dict_prev_game: Mapping[str, Any],
                       tuple_actions: Sequence[Any]):
    '''
    user_cur_game_data: NOTE - values will be updated
    '''

    pass

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    user_game_data.data[Exp1UserData.GAME_DONE] = True

    # update score
    game = user_game_data.get_game_ref()
    user_game_data.data[Exp1UserData.SCORE] = game.current_step

  def _get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
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

    obj = self._get_instruction_objs(user_data)
    dict_objs[obj.name] = obj

    disable_status = self._get_action_btn_disable_state(user_data, dict_game)
    objs = self._get_btn_actions(dict_game, *disable_status)
    for obj in objs:
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    return None

  def _get_animations(self, dict_prev_game: Mapping[str, Any],
                      dict_cur_game: Mapping[str, Any]):

    return []

  def action_event(self, user_game_data: Exp1UserData, clicked_btn: str):
    '''
    user_game_data: NOTE - values will be updated
    '''
    action = None
    if clicked_btn == self.CN_STAY:
      action = ActionCN.STAY
    elif clicked_btn == self.CN_UP:
      action = ActionCN.MOVE_UP
    elif clicked_btn == self.CN_DOWN:
      action = ActionCN.MOVE_DOWN
    elif clicked_btn == self.CN_LEFT:
      action = ActionCN.MOVE_LEFT
    elif clicked_btn == self.CN_RIGHT:
      action = ActionCN.MOVE_RIGHT
    elif clicked_btn == self.CN_PICKUP:
      action = ActionCN.PICKUP
    elif clicked_btn == self.CN_HANDOVER:
      action = ActionCN.HANDOVER

    game = user_game_data.get_game_ref()
    # should not happen
    assert action is not None
    assert not game.is_finished()

    game.event_input(self._CN, action, None)
    game.event_input(self._SN, ActionSN.STAY, None)
    game.event_input(self._AS, ActionAS.STAY, None)

    # take actions
    map_agent2action = game.get_joint_action()
    game.take_a_step(map_agent2action)

    return ([], [], [], game.is_finished())

  def _game_overlay(self, game_env,
                    user_data: Exp1UserData) -> List[co.DrawingObject]:

    def coord_2_canvas(coord_x, coord_y):
      x = int(self.GAME_LEFT + coord_x * self.GAME_WIDTH)
      y = int(self.GAME_TOP + coord_y * self.GAME_HEIGHT)
      return (x, y)

    def size_2_canvas(width, height):
      w = int(width * self.GAME_WIDTH)
      h = int(height * self.GAME_HEIGHT)
      return (w, h)

    overlay_obs = []

    return overlay_obs

  def _game_overlay_names(self, game_env, user_data: Exp1UserData) -> List:
    overlay_names = []
    return overlay_names

  def _game_scene(self,
                  game_env,
                  user_data: Exp1UserData,
                  include_background: bool = True) -> List[co.DrawingObject]:

    game_ltwh = (self.GAME_LEFT, self.GAME_TOP, self.GAME_WIDTH,
                 self.GAME_HEIGHT)
    return tooldelivery_game_scene(game_env, game_ltwh, include_background)

  def _game_scene_names(self, game_env, user_data: Exp1UserData) -> List:

    def is_visible(img_name):
      return True

    return tooldelivery_game_scene_names(game_env, is_visible)

  def latent2selbtn(self, latent):
    return "latent" + str(latent)

  def selbtn2latent(self, sel_btn_name):
    if sel_btn_name[:6] == "latent":
      return int(sel_btn_name[6:])

    return None

  def is_sel_latent_btn(self, sel_btn_name):
    return sel_btn_name[:6] == "latent"
