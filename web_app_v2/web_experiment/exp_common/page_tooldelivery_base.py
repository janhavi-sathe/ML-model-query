from typing import Mapping, Any, Sequence, List
import copy
import numpy as np

from ai_coach_domain.tooldelivery.environment import RequestEnvironment
from ai_coach_domain.tooldelivery.tooldelivery_v3_mdp import ToolDeliveryMDP_V3
from ai_coach_domain.tooldelivery.tooldelivery_v3_policy import ToolDeliveryPolicy_V3
import ai_coach_domain.tooldelivery.tooldelivery_v3_state_action as T3SA
from ai_coach_domain.tooldelivery.simulator import ToolDeliverySimulator

from ai_coach_domain.agent import InteractiveAgent
from web_experiment.exp_common.helper import tooldelivery_game_scene, tooldelivery_game_scene_names
import web_experiment.exp_common.canvas_objects as co
from web_experiment.models import db, User
from web_experiment.define import EDomainType
from web_experiment.exp_common.page_base import ExperimentPageBase, Exp1UserData
from web_experiment.exp_common.helper import (get_file_name,
                                              store_user_label_locally)

class ToolDeliveryGamePageBase(ExperimentPageBase):
  ACTION_BUTTONS = ["next"]

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

    if clicked_btn == self.ACTION_BUTTONS[0]:
      _, _, _, done = self.action_event(user_game_data, clicked_btn)
      if done:
        self._on_game_finished(user_game_data)
      return

    return super().button_clicked(user_game_data, clicked_btn)

  def _get_instruction(self, user_game_data: Exp1UserData):
    return ("Click the button to progress through the simulation.")

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
    return False, False, False

  def _get_btn_actions(
      self,
      game_env: Mapping[Any, Any],
      disable_move: bool = False,
      disable_stay: bool = False,
      disable_rescue: bool = False) -> Sequence[co.DrawingObject]:

    x_ctrl_cen = int(self.GAME_RIGHT + (co.CANVAS_WIDTH - self.GAME_RIGHT) / 2)
    y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
    x_joy_cen = int(x_ctrl_cen - 75)
    ctrl_origin = np.array([x_joy_cen, y_ctrl_cen])

    arrow_width = 30
    font_size = 18

    list_buttons = []

    offset = 30

    for dir in [np.array([1, 0])]:
      origin = ctrl_origin + dir * offset
      origin = (int(origin[0]), int(origin[1]))
      direction = (int(dir[0]), int(dir[1]))
      btn_obj = co.ThickArrow(self.ACTION_BUTTONS[0],
                                  origin,
                                  direction,
                                  arrow_width,
                                  disable=disable_move)
      list_buttons.append(btn_obj)

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
    game = user_game_data.get_game_ref()

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
