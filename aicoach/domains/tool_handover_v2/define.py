from enum import Enum
from TMM.models.mdp import ActionSpace, StateSpace


class EventType(Enum):
  Action = 0
  Set_Latent = 1


class PatientVital(Enum):
  Stable = 0
  Unstable = 1


# NOTE: this will also be nurse's mental state.
#       some are used to represent tools as well.
class Requirement(Enum):
  Nurse_Assist = 0
  Hand_Only = 1
  Antiseptic_Solution = 2
  Lab_Test = 3
  Scalpel = 4
  Sternal_Saw = 5
  Clamp = 6
  Needle = 7
  EVH_system = 8
  Tourniquet = 9
  Dressing = 10
  Forceps = 11
  Suture = 12


class ToolLocation(Enum):
  Nurse = 0
  Surgeon = 1
  Original = 2


class PickupLocation(Enum):
  Quadrant1 = 1
  Quadrant2 = 2
  Quadrant3 = 3
  Quadrant4 = 4


class NurseDirection(Enum):
  Up = 0
  Left = 1
  Down = 2
  Right = 3


class SurgeonAction(Enum):
  Stay = 0
  Proceed = 1
  Exchange_Tool = 2  # receive/exchange/handover tool
  Tell_Requirement = 3


class NurseAction(Enum):  # with tool
  Stay = 0
  Rotate_Left = 1
  Rotate_Right = 2
  Rotate_180 = 3
  Move_Forward = 4
  Ask_Requirement = 5
  Assist = 6
  PickUp_Drop = 7  # along with Pickup_Location


class AnesthesiaAction(Enum):
  Stay = 0
  Proceed = 1


class PerfusionAction(Enum):
  Stay = 0
  Proceed = 1


NURSE_ACTIONSPACE = ActionSpace([(NurseAction(idx), None)
                                 for idx in range(len(NurseAction) - 1)] +
                                [(NurseAction.PickUp_Drop, loc)
                                 for loc in PickupLocation])

POSSIBLE_TOOLS = list(Requirement)[1:]


def can_exchange_tool(nurse_pos, nurse_dir, surgeon_pos):
  return (nurse_pos[0] == surgeon_pos[0] - 1 and nurse_pos[1] == surgeon_pos[1]
          and nurse_dir in [NurseDirection.Down, NurseDirection.Right])


def get_target_pos(nurse_pos, nurse_dir):
  target_pos = nurse_pos
  if nurse_dir == NurseDirection.Up:
    target_pos = (nurse_pos[0], nurse_pos[1] - 1)
  elif nurse_dir == NurseDirection.Left:
    target_pos = (nurse_pos[0] - 1, nurse_pos[1])
  elif nurse_dir == NurseDirection.Down:
    target_pos = (nurse_pos[0], nurse_pos[1] + 1)
  elif nurse_dir == NurseDirection.Right:
    target_pos = (nurse_pos[0] + 1, nurse_pos[1])
  else:
    raise ValueError("Invalid nurse direction")

  return target_pos
