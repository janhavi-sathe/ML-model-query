from typing import Hashable, Tuple
from TMM.domains.box_push import EventType
from TMM.domains.box_push.simulator import BoxPushSimulator
from .transition import transition_alone_and_together


class BoxPushSimulator_AloneOrTogether(BoxPushSimulator):

  def __init__(
      self,
      id: Hashable,
      tuple_action_when_none: Tuple = (EventType.STAY, EventType.STAY)
  ) -> None:
    super().__init__(id, tuple_action_when_none=tuple_action_when_none)

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition_alone_and_together(self.box_states, self.a1_pos,
                                         self.a2_pos, a1_action, a2_action,
                                         self.boxes, self.goals, self.walls,
                                         self.drops, self.x_grid, self.y_grid)
