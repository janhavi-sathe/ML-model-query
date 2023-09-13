import abc
from typing import Mapping, Any
from aic_domain.simulator import Simulator
import web_experiment.exp_common.canvas_objects as co
from web_experiment.define import EDomainType
from .page_base import Exp1UserData, ExperimentPageBase


class SurgeryUserData(Exp1UserData):
  SELECT_IT = "select_it"
  SELECT_MH = "select_mh"
  S_HANDOVER = "s_handover"

  def __init__(self, user) -> None:
    super().__init__(user)
    self.data[SurgeryUserData.SELECT_IT] = False
    self.data[SurgeryUserData.SELECT_MH] = False
    self.data[SurgeryUserData.S_HANDOVER] = False


class SurgeryPageBase(ExperimentPageBase):
  GAME_LEFT = 0
  GAME_TOP = int(co.CANVAS_HEIGHT * 1 / 6)
  GAME_HEIGHT = int(co.CANVAS_HEIGHT * 5 / 6)
  GAME_WIDTH = GAME_HEIGHT
  GAME_RIGHT = GAME_LEFT + GAME_WIDTH
  GAME_BOTTOM = GAME_TOP + GAME_HEIGHT

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
