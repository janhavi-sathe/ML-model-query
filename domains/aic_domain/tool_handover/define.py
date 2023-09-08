from enum import Enum
from ai_coach_core.utils.mdp_utils import ActionSpace, StateSpace


class PatientState(Enum):
  Stable = 0
  Bad = 1


class SurgeonSight(Enum):
  Patient = 0
  Table = 1


class SurgicalStep(Enum):
  Step_0 = 0
  Step_1 = 1
  Step_2 = 2
  Step_3 = 3
  Step_4 = 4


SURGICAL_STEP_TERMINAL = "Surgery Done"


class Tool_Type(Enum):
  Tool_1 = 0
  Tool_2 = 1
  Tool_3 = 2
  Tool_4 = 3


class Tool_Location(Enum):
  Nurse = 0
  Surgeon = 1
  Table_1 = 2
  Table_2 = 3
  Table_3 = 4
  Table_4 = 5


NURSE_HAND_POSITIONS = [
    Tool_Location(idx) for idx in range(1, len(Tool_Location))
]

TOOL_FOR_CUR_STEP = [None] + list(Tool_Type)


# mental state: no_need_tool / tool1_in_need / tool2_in_need / ...
class MentalState(Enum):
  No_Need_Tool = 0
  Need_Tool = 1  # along with tool_type


MENTAL_STATESPACE = StateSpace([(MentalState.No_Need_Tool, None)] +
                               [(MentalState.Need_Tool, item)
                                for item in Tool_Type])

# -- Surgeon --
# patient state + surgical step --> mental state (tool)
# mental state (tool) != tool_on_hand  --> gesture
# not sight_patient --> next_step not possible
# table_sight --> don't know patient state
# stay + tool don't need

# -- Nurse --
# (gesture, any_nurse_action) + patient_state --> mental state (tool)
# mental state (tool) + (not gesture, handover) + tool_location --> mental state (no tool)
# table_sight + gesture --> correct mental state (tool)


class SurgeonAction(Enum):
  Stay = 0  # receive tool
  Change_View = 1
  Next_Step = 2  # success rate depends on the tool
  Handover = 3  # receive/exchange tool
  Gesture_Tool = 4  # tool request signal
  Indicate_Tool = 5  # along with tool_type


class NurseAction(Enum):  # with tool
  Stay = 0
  PickUp = 1  # handover
  Drop = 2  # handover
  Move_hand = 3  # along with hand position


SURGEON_ACTIONSPACE = ActionSpace([(SurgeonAction(idx), None)
                                   for idx in range(5)] +
                                  [(SurgeonAction.Indicate_Tool, item)
                                   for item in Tool_Type])

NURSE_ACTIONSPACE = ActionSpace([(NurseAction(idx), None) for idx in range(3)] +
                                [(NurseAction.Move_hand, pos)
                                 for pos in NURSE_HAND_POSITIONS])
