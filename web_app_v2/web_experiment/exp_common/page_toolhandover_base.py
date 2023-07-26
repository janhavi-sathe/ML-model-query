from typing import Mapping, Any, Sequence, List
import copy
import numpy as np

from ai_coach_domain.agent import InteractiveAgent
from ai_coach_domain.tool_handover.mdp import MDP_ToolHandover
from ai_coach_domain.tool_handover.simulator import ToolHandoverSimulator
import ai_coach_domain.tool_handover.define as tho

from ai_coach_domain.agent import InteractiveAgent
from web_experiment.exp_common.helper import toolhandover_game_scene, toolhandover_game_scene_names
import web_experiment.exp_common.canvas_objects as co
from web_experiment.models import db, User
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import ExperimentPageBase, Exp1UserData
from web_experiment.exp_common.helper import (get_file_name,
                                              store_user_label_locally)

class ToolHandoverGamePageBase(ExperimentPageBase):
  S_STAY = tho.SurgeonAction.Stay.name
  S_CV = tho.SurgeonAction.Change_View.name
  S_NS = tho.SurgeonAction.Next_Step.name
  S_HANDOVER = tho.SurgeonAction.Handover.name
  S_GT = tho.SurgeonAction.Gesture_Tool.name
  S_IT = tho.SurgeonAction.Indicate_Tool.name

  N_STAY = tho.NurseAction.Stay.name
  N_PU = tho.NurseAction.PickUp.name
  N_DROP = tho.NurseAction.Drop.name
  N_MH = tho.NurseAction.Move_hand.name

  ACTION_BUTTONS = [S_STAY, S_CV, S_NS, S_HANDOVER, S_GT, S_IT,
                    # N_STAY, N_PU, N_DROP, N_MH
                    ]
  # ACTION_BUTTONS = ["next"]

  def __init__(self,
               manual_latent_selection,
               game_map,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(True, True, True, EDomainType.ToolHandover)
    self._MANUAL_SELECTION = manual_latent_selection
    self._GAME_MAP = game_map

    self._PROMPT_ON_CHANGE = prompt_on_change
    self._PROMPT_FREQ = prompt_freq
    self._AUTO_PROMPT = auto_prompt

    self._S = ToolHandoverSimulator.Surgeon
    self._N = ToolHandoverSimulator.Nurse

  def init_user_data(self, user_game_data: Exp1UserData):
    user_game_data.data[Exp1UserData.GAME_DONE] = False

    game = user_game_data.get_game_ref()
    if game is None:
      game = ToolHandoverSimulator()

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
    return ("Control the CN with the buttons to progress through the simulation.")

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

    it_ok = True
    surgeon_sight = game_env["surgeon_sight"]
    
    if surgeon_sight == tho.SurgeonSight.Patient:
      it_ok = False

    return False, False, False, not it_ok, False, False

  def _get_btn_actions(
      self,
      game_env: Mapping[Any, Any],
      disable_cv: bool = False,
      disable_ns: bool = False,
      disable_gt: bool = False,
      disable_it: bool = False, 
      disable_stay: bool = False,
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
                              disable=disable_stay, name=self.S_STAY)
    
    font_size = 20
    btn_cv = co.ButtonRect(
        self.S_CV,
        (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen - int(ctrl_btn_w * 1.8)),
        (ctrl_btn_w * 2, ctrl_btn_w),
        font_size,
        "Change View",
        disable=disable_cv)
    btn_ns = co.ButtonRect(
        self.S_NS,
        (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen - int(ctrl_btn_w * 0.6)),
        (ctrl_btn_w * 2, ctrl_btn_w),
        font_size,
        "Next Step",
        disable=disable_ns)
    btn_handover = co.ButtonRect(
        self.S_HANDOVER,
        (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen + int(ctrl_btn_w * 0.6)),
        (ctrl_btn_w * 2, ctrl_btn_w),
        font_size,
        "Next Step",
        disable=disable_handover)
    btn_gt = co.ButtonRect(
        self.S_GT,
        (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen + int(ctrl_btn_w * 1.8)),
        (ctrl_btn_w * 2, ctrl_btn_w),
        font_size,
        "Gather Tool",
        disable=disable_gt)
    btn_it = co.ButtonRect(
        self.S_IT,
        (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen + int(ctrl_btn_w * 3.0)),
        (ctrl_btn_w * 2, ctrl_btn_w),
        font_size,
        "Indicate Tool",
        disable=disable_it)
  
    list_buttons = [btn_stay, btn_cv, btn_ns, btn_handover, btn_gt, btn_it]
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
    if clicked_btn == self.S_STAY:
      action = tho.SurgeonAction.Stay
    elif clicked_btn == self.S_CV:
      action = tho.SurgeonAction.Change_View
    elif clicked_btn == self.S_NS:
      action = tho.SurgeonAction.Next_Step
    elif clicked_btn == self.S_HANDOVER:
      action = tho.SurgeonAction.Handover
    elif clicked_btn == self.S_GT:
      action = tho.SurgeonAction.Gesture_Tool
    elif clicked_btn == self.S_IT:
      action = tho.SurgeonAction.Indicate_Tool
    
    game = user_game_data.get_game_ref()
    # should not happen
    assert action is not None
    assert not game.is_finished()

    game.event_input(self._S, action, None)
    game.event_input(self._N, tho.NurseAction.STAY, None)

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
    return toolhandover_game_scene(game_env, game_ltwh, include_background)

  def _game_scene_names(self, game_env, user_data: Exp1UserData) -> List:
    def is_visible(img_name):
      return True

    return toolhandover_game_scene_names(game_env, is_visible)

  def latent2selbtn(self, latent):
    return "latent" + str(latent)

  def selbtn2latent(self, sel_btn_name):
    if sel_btn_name[:6] == "latent":
      return int(sel_btn_name[6:])

    return None

  def is_sel_latent_btn(self, sel_btn_name):
    return sel_btn_name[:6] == "latent"
