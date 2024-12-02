from typing import Sequence, Tuple, Any, Union, Mapping
import aicoach.domains.tool_handover_v2.define as tho
from copy import deepcopy


def tool_handover_transition_v2(
    patient_vital: tho.PatientVital, nurse_dir: tho.NurseDirection,
    nurse_pos: Tuple[int, int], nurse_tool: tho.Requirement,
    surgeon_tool: tho.Requirement, surgeon_ready: bool, anes_ready: bool,
    perf_ready: bool, cur_step: int, cur_requirement: tho.Requirement,
    nurse_asked: bool, nurse_action, surgeon_action, anes_action, perf_action,
    surgeon_pos, nurse_possible_pos, table_blocks: Sequence,
    surgical_steps: Sequence, tool_table_zone: Mapping, **kwargs):

  if cur_step >= len(surgical_steps):
    return [(1.0, patient_vital, nurse_dir, nurse_pos, nurse_tool, surgeon_tool,
             surgeon_ready, anes_ready, perf_ready, cur_step, cur_requirement,
             nurse_asked)]

  # for multiple use
  target_pos = tho.get_target_pos(nurse_pos, nurse_dir)

  # nurse direction
  dist_nurse_dir = []
  new_nurse_dir = nurse_dir
  if nurse_action[0] == tho.NurseAction.Rotate_Left:
    new_nurse_dir = tho.NurseDirection((nurse_dir.value + 1) % 4)
  elif nurse_action[0] == tho.NurseAction.Rotate_Right:
    new_nurse_dir = tho.NurseDirection((nurse_dir.value + 3) % 4)
  elif nurse_action[0] == tho.NurseAction.Rotate_180:
    new_nurse_dir = tho.NurseDirection((nurse_dir.value + 2) % 4)
  dist_nurse_dir.append((1.0, new_nurse_dir))

  # nurse pos
  dist_nurse_pos = []
  new_nurse_pos = nurse_pos
  if nurse_action[0] == tho.NurseAction.Move_Forward:
    if target_pos in nurse_possible_pos:
      new_nurse_pos = target_pos
  dist_nurse_pos.append((1.0, new_nurse_pos))

  # nurse asked
  dist_nurse_asked = []
  new_nurse_asked = nurse_asked
  if nurse_action[0] == tho.NurseAction.Ask_Requirement:
    new_nurse_asked = True
  if surgeon_action == tho.SurgeonAction.Tell_Requirement:
    new_nurse_asked = False
  dist_nurse_asked.append((1.0, new_nurse_asked))

  # nurse and surgeon tool
  dist_nurse_surgeon_tool = []
  new_nurse_surgeon_tool = [nurse_tool, surgeon_tool]
  if (tho.can_exchange_tool(nurse_pos, nurse_dir, surgeon_pos)
      and surgeon_action == tho.SurgeonAction.Exchange_Tool):
    new_nurse_surgeon_tool = [surgeon_tool, nurse_tool]
  elif nurse_action[0] == tho.NurseAction.PickUp_Drop:
    # quadrant = (nurse_dir.value + nurse_action[1].value) % 4 + 1
    quadrant = nurse_action[1].value
    if target_pos in table_blocks:
      idx = table_blocks.index(target_pos)
      for tool, table_zone in tool_table_zone.items():
        if table_zone[0] == idx and table_zone[1] == quadrant:
          if (nurse_tool == tho.Requirement.Hand_Only
              and tool != surgeon_tool):  # pickup
            new_nurse_surgeon_tool[0] = tool
          elif tool == nurse_tool:  # drop
            new_nurse_surgeon_tool[0] = tho.Requirement.Hand_Only
          else:  # do nothing (already holding another tool)
            pass
  dist_nurse_surgeon_tool.append((1.0, new_nurse_surgeon_tool))

  # anes ready_state
  P_ANES_RDY = 0.8
  dist_anes_ready = [(1.0, anes_ready)]
  if anes_ready is False and anes_action == tho.AnesthesiaAction.Proceed:
    dist_anes_ready = [(P_ANES_RDY, True), (1 - P_ANES_RDY, anes_ready)]

  # perf ready_state
  P_PERF_RDY = 0.8
  dist_perf_ready = [(1.0, perf_ready)]
  if perf_ready is False and perf_action == tho.PerfusionAction.Proceed:
    dist_perf_ready = [(P_PERF_RDY, True), (1 - P_PERF_RDY, perf_ready)]

  # pateint vital
  dist_pat_vital_next_req = [(1, [patient_vital, cur_requirement])]
  if (patient_vital == tho.PatientVital.Unstable
      and surgeon_action == tho.SurgeonAction.Proceed):
    P_GET_STABLE = 0.8
    appropriate = False
    if cur_requirement == tho.Requirement.Nurse_Assist:
      if (tho.can_exchange_tool(nurse_pos, nurse_dir, surgeon_pos)
          and nurse_action[0] == tho.NurseAction.Assist):
        appropriate = True
    else:
      if surgeon_tool == cur_requirement:
        appropriate = True

    if appropriate:
      dist_pat_vital_next_req = [(1 - P_GET_STABLE,
                                  [patient_vital, cur_requirement])]
      for req, p_r in surgical_steps[cur_step][tho.PatientVital.Stable]:
        prop = P_GET_STABLE * p_r
        dist_pat_vital_next_req.append((prop, [tho.PatientVital.Stable, req]))

  elif (patient_vital == tho.PatientVital.Stable and surgeon_ready is False
        and surgeon_action != tho.SurgeonAction.Proceed):
    if cur_step != 0:
      P_GET_UNSTABLE = 0.01
      dist_pat_vital_next_req = [(1 - P_GET_UNSTABLE,
                                  [patient_vital, cur_requirement])]
      for req, p_r in surgical_steps[cur_step][tho.PatientVital.Unstable]:
        prop = P_GET_UNSTABLE * p_r
        dist_pat_vital_next_req.append((prop, [tho.PatientVital.Unstable, req]))

  # surgeon ready_state
  P_SURG_RDY = 0.8
  dist_surg_ready = [(1, surgeon_ready)]
  if (patient_vital == tho.PatientVital.Stable and surgeon_ready is False
      and surgeon_action == tho.SurgeonAction.Proceed):
    appropriate = False
    if cur_requirement == tho.Requirement.Nurse_Assist:
      if (tho.can_exchange_tool(nurse_pos, nurse_dir, surgeon_pos)
          and nurse_action[0] == tho.NurseAction.Assist):
        appropriate = True
    else:
      if surgeon_tool == cur_requirement:
        appropriate = True

    if appropriate:
      dist_surg_ready = [(1 - P_SURG_RDY, surgeon_ready), (P_SURG_RDY, True)]

  # merge ready, cur_step, cur_requirement, vital
  list_p_ready_vital_req_step = []
  for p_surg_rdy, s_surg_rdy in dist_surg_ready:
    for p_anes_rdy, s_anes_rdy in dist_anes_ready:
      for p_perf_rdy, s_perf_rdy in dist_perf_ready:
        p_rdy = p_surg_rdy * p_perf_rdy * p_anes_rdy
        if s_surg_rdy and s_perf_rdy and s_anes_rdy:
          new_step = cur_step + 1
          if new_step >= len(surgical_steps):
            list_p_ready_vital_req_step.append(
                (p_rdy, patient_vital, surgeon_ready, anes_ready, perf_ready,
                 new_step, cur_requirement))  # terminal state
          else:
            for req, p_r in surgical_steps[new_step][tho.PatientVital.Stable]:
              prop = p_rdy * p_r
              if prop > 0:
                list_p_ready_vital_req_step.append(
                    (prop, tho.PatientVital.Stable, False, False, False,
                     new_step, req))
        else:
          for p_vital_req, s_vital_req in dist_pat_vital_next_req:
            prop = p_rdy * p_vital_req
            if prop > 0:
              s_vital, s_req = s_vital_req
              list_p_ready_vital_req_step.append(
                  (prop, s_vital, s_surg_rdy, s_anes_rdy, s_perf_rdy, cur_step,
                   s_req))

  # merge all
  list_p_state = []
  for p_n_dir, s_n_dir in dist_nurse_dir:
    for p_n_pos, s_n_pos in dist_nurse_pos:
      for p_n_ask, s_n_ask in dist_nurse_asked:
        for p_n_surgeon_tool, s_n_surgeon_tool in dist_nurse_surgeon_tool:
          s_n_tool, s_s_tool = s_n_surgeon_tool
          for item in list_p_ready_vital_req_step:
            p_rest, s_vital, s_s_rdy, s_a_rdy, s_p_rdy, s_step, s_req = item
            prop = p_n_dir * p_n_pos * p_n_ask * p_n_surgeon_tool * p_rest
            if prop > 0:
              list_p_state.append(
                  (prop, s_vital, s_n_dir, s_n_pos, s_n_tool, s_s_tool, s_s_rdy,
                   s_a_rdy, s_p_rdy, s_step, s_req, s_n_ask))

  return list_p_state
