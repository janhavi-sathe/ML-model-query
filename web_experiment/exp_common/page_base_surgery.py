from typing import Mapping
import web_experiment.exp_common.canvas_objects as co
from .page_base import Exp1UserData, ExperimentPageBase


class SurgeryUserData(Exp1UserData):
  SELECT_MH = "select_mh"
  DIALOGS = "dialogs"

  def __init__(self, user) -> None:
    super().__init__(user)
    self.data[SurgeryUserData.SELECT_MH] = False
    # dialog item should be (agent, text) format
    self.data[SurgeryUserData.DIALOGS] = []


class SurgeryPageBase(ExperimentPageBase):
  GAME_LEFT = 0
  GAME_TOP = int(co.CANVAS_HEIGHT * 1 / 6)
  GAME_HEIGHT = int(co.CANVAS_HEIGHT * 5 / 6)
  GAME_WIDTH = GAME_HEIGHT
  GAME_RIGHT = GAME_LEFT + GAME_WIDTH
  GAME_BOTTOM = GAME_TOP + GAME_HEIGHT
  INSTRUCTION_BORDER = "instruction_border"

  def _get_drawing_order(self, user_game_data: SurgeryUserData = None):
    drawing_order = []
    if self._SHOW_BORDER:
      drawing_order.append(self.GAME_BORDER)
      drawing_order.append(self.INSTRUCTION_BORDER)

    return drawing_order

  def _get_init_drawing_objects(
      self, user_data: SurgeryUserData) -> Mapping[str, co.DrawingObject]:

    dict_objs = {}
    if self._SHOW_BORDER:
      dict_objs[self.GAME_BORDER] = co.Rectangle(
          self.GAME_BORDER, (self.GAME_LEFT, self.GAME_TOP),
          (self.GAME_WIDTH, self.GAME_HEIGHT),
          fill=False,
          border=True,
          linewidth=3)
      dict_objs[self.INSTRUCTION_BORDER] = co.Rectangle(
          self.INSTRUCTION_BORDER, (0, 0), (self.GAME_WIDTH, self.GAME_TOP),
          fill=False,
          border=True,
          linewidth=3)

    if self._SHOW_INSTRUCTION:
      obj = self._get_instruction_objs(user_data)
      dict_objs[obj.name] = obj

    if self._SHOW_SCORE:
      obj = self._get_score_obj(user_data)
      dict_objs[obj.name] = obj

    return dict_objs

  def _get_spotlight(self, x_cen, y_cen, radius):
    outer_ltwh = (0, 0, co.CANVAS_WIDTH, co.CANVAS_HEIGHT)

    margin = 5
    pos = (margin, margin)
    size = (self.GAME_WIDTH - 2 * margin, self.GAME_TOP - 2 * margin)
    return co.ClippedRectangle(self.SPOTLIGHT,
                               outer_ltwh,
                               list_circle=[(x_cen, y_cen, radius)],
                               list_rect=[(*pos, *size)])

  def _get_instruction_objs(self, user_data: SurgeryUserData):
    margin = 10
    pos = (margin, margin)
    width = self.GAME_WIDTH - 2 * margin
    text_instr = co.TextObject(self.TEXT_INSTRUCTION, pos, width, 18,
                               self._get_instruction(user_data))

    return text_instr
