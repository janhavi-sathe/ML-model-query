from typing import Sequence, Tuple, Any, Union
import aic_domain.tool_handover.define as tho
from copy import deepcopy


def tool_handover_transition(
    patient_state: tho.PatientState, surgeon_sight: tho.SurgeonSight,
    surgical_step: tho.SurgicalStep, nurse_hand: tho.Tool_Location,
    tool_for_now: Union[None, tho.Tool_Type],
    list_tool_locations: Sequence[tho.Tool_Location],
    list_tool_types: Sequence[tho.Tool_Type],
    surgeon_action: Tuple[tho.SurgeonAction,
                          Any], nurse_action: Tuple[tho.NurseAction, Any]):

  assert (len(list_tool_types) == len(list_tool_locations))

  # terminal state
  if surgical_step == tho.SURGICAL_STEP_TERMINAL:
    return [(1.0, patient_state, surgeon_sight, surgical_step, nurse_hand,
             tool_for_now, list_tool_locations)]

  P_STABLE2BAD = 0.3
  P_BAD2STABLE = 0.3

  # patient state changes randomly
  dist_patient = []
  if patient_state == tho.PatientState.Bad:
    dist_patient.append((P_BAD2STABLE, tho.PatientState.Stable))
    dist_patient.append((1 - P_BAD2STABLE, patient_state))
  else:
    dist_patient.append((P_STABLE2BAD, tho.PatientState.Bad))
    dist_patient.append((1 - P_STABLE2BAD, patient_state))

  # surgeon sight changes when surgeon action is Change_view
  P_CHANGEVIEW = 1.0
  dist_surgeon_sight = []
  if surgeon_action[0] == tho.SurgeonAction.Change_View:
    if surgeon_sight == tho.SurgeonSight.Table:
      dist_surgeon_sight.append((P_CHANGEVIEW, tho.SurgeonSight.Patient))
      dist_surgeon_sight.append((1 - P_CHANGEVIEW, surgeon_sight))
    else:  # SurgeonSight.Patient
      dist_surgeon_sight.append((P_CHANGEVIEW, tho.SurgeonSight.Table))
      dist_surgeon_sight.append((1 - P_CHANGEVIEW, surgeon_sight))
  else:
    dist_surgeon_sight.append((1.0, surgeon_sight))

  # get tool at location
  def get_tool_idx(location, tool_locations):
    for idx in range(len(tool_locations)):
      if tool_locations[idx] == location:
        return idx
    return None

  # surgical step changes when surgeon perform next_step action
  # TODO: if surgeon is not holding the correct tool, make the task not proceed to next step.
  # TODO: correct tools depend on patient state and surgical step
  P_PROCEED = 0.3
  tidx_with_surgeon = get_tool_idx(tho.Tool_Location.Surgeon,
                                   list_tool_locations)
  if tidx_with_surgeon is None:
    surgeon_tool = None
  else:
    surgeon_tool = list_tool_types[tidx_with_surgeon]

  dist_surgical_step = []
  if (surgeon_action[0] == tho.SurgeonAction.Next_Step
      and surgeon_tool == tool_for_now):
    if surgical_step.value < len(tho.SurgicalStep):
      next_surgical_step = tho.SURGICAL_STEP_TERMINAL
      if surgical_step.value < len(tho.SurgicalStep) - 1:
        next_surgical_step = tho.SurgicalStep(surgical_step.value + 1)

      dist_surgical_step.append((P_PROCEED, next_surgical_step))
      dist_surgical_step.append((1 - P_PROCEED, surgical_step))
    else:
      dist_surgical_step.append((1.0, surgical_step))
  else:
    dist_surgical_step.append((1.0, surgical_step))

  # position of nurse hand changes when nurse take move_hand action
  P_HANDMOVE = 1.0
  dist_nurse_hand = []
  if nurse_action[0] == tho.NurseAction.Move_hand:
    dist_nurse_hand.append((P_HANDMOVE, nurse_action[1]))
    dist_nurse_hand.append((1 - P_HANDMOVE, nurse_hand))
  else:
    dist_nurse_hand.append((1.0, nurse_hand))

  # tool state changes according to nurse pick-up, nurse drop and surgeon handover actions
  P_TOOLMOVE = 1.0
  dist_tool_state = []
  tidx_around_nurse_hand = get_tool_idx(nurse_hand, list_tool_locations)
  tidx_with_nurse = get_tool_idx(tho.Tool_Location.Nurse, list_tool_locations)

  is_moved = False
  new_list_tool_locs = deepcopy(list_tool_locations)
  if (nurse_hand != tho.Tool_Location.Surgeon
      or surgeon_action[0] == tho.SurgeonAction.Handover):
    if nurse_action[0] == tho.NurseAction.PickUp:
      if tidx_with_nurse is None and tidx_around_nurse_hand is not None:
        new_list_tool_locs[tidx_around_nurse_hand] = tho.Tool_Location.Nurse
        is_moved = True
    elif nurse_action[0] == tho.NurseAction.Drop:
      if tidx_with_nurse is not None and tidx_around_nurse_hand is None:
        new_list_tool_locs[tidx_with_nurse] = nurse_hand
        is_moved = True

  if is_moved:
    dist_tool_state.append((P_TOOLMOVE, new_list_tool_locs))
    dist_tool_state.append((1 - P_TOOLMOVE, list_tool_locations))
  else:
    dist_tool_state.append((1.0, list_tool_locations))

  list_p_state = []
  for p_pat, s_pat in dist_patient:
    for p_sight, s_sight in dist_surgeon_sight:
      for p_step, s_step in dist_surgical_step:
        for p_hand, s_hand in dist_nurse_hand:
          for p_tool, s_tool in dist_tool_state:
            prop = p_pat * p_sight * p_step * p_hand * p_tool
            if prop > 0:
              # the tool for now is different depending on s_pat and s_step
              if s_step == tho.SURGICAL_STEP_TERMINAL:
                idx_tool = 0
              else:
                idx_tool = s_step.value * len(tho.PatientState) + s_pat.value
                idx_tool = idx_tool % len(tho.TOOL_FOR_CUR_STEP)
              s_tool_fornow = tho.TOOL_FOR_CUR_STEP[idx_tool]

              list_p_state.append(
                  (prop, s_pat, s_sight, s_step, s_hand, s_tool_fornow, s_tool))

  return list_p_state
