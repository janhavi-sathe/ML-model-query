from typing import Mapping, Any, List, Tuple, Callable, Sequence
import web_experiment.exp_common.canvas_objects as co
import aic_domain.tooldelivery.tooldelivery_v3_state_action as T3SA
import aic_domain.tool_handover.define as tho


def tooldelivery_game_scene(
    game_env: Mapping[str, Any],
    game_ltwh: Tuple[int, int, int, int],
    include_background: bool = True,
    cb_is_visible: Callable[[co.DrawingObject], bool] = None
) -> List[co.DrawingObject]:
  game_left, game_top, game_width, game_height = game_ltwh

  # the / 10 and / 5 are hardcoded as 10 x units and 5 y units made up the canvas in the standalone app
  def coord_2_canvas(coord_x, coord_y):
    x = int(game_left + coord_x / 10 * game_width)
    y = int(game_top + coord_y / 5 * game_height)
    return (x, y)

  def size_2_canvas(width, height):
    w = int(width / 10 * game_width)
    h = int(height / 5 * game_height)
    return (w, h)

  # place_w = 0.12
  # place_h = 0.12

  game_objs = []

  # game_objs.append(co.Rectangle("Background", (game_left, game_top), (game_width, game_height), "blue", "green", 0.5))

  if include_background:
    for idx, coord in enumerate(game_env["Walls"]):
      x, y = coord
      obj = co.LineSegment(f"Wall{idx}", coord_2_canvas(x, y),
                           coord_2_canvas(x, y + 1), 1, "black")
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

  def draw_pos_size_obj(pos_size, obj_name):
    x, y, w, h = pos_size

    obj = co.GameObject(obj_name, coord_2_canvas(x, y), size_2_canvas(w, h), 0,
                        obj_name)
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  SN_pos_size = game_env["SN_pos_size"]
  AS_pos_size = game_env["AS_pos_size"]
  Table_pos_size = game_env["Table_pos_size"]
  Patient_pos_size = game_env["Patient_pos_size"]
  Perfusionist_pos_size = game_env["Perfusionist_pos_size"]
  Anesthesiologist_pos_size = game_env["Anesthesiologist_pos_size"]
  Cabinet_pos_size = game_env["Cabinet_pos_size"]
  Storage_pos_size = game_env["Storage_pos_size"]
  Handover_pos_size = game_env["Handover_pos_size"]
  draw_pos_size_obj(SN_pos_size, co.IMG_SCRUB)
  draw_pos_size_obj(AS_pos_size, co.IMG_SURGEON)
  draw_pos_size_obj(Table_pos_size, co.IMG_TABLE)
  draw_pos_size_obj(Patient_pos_size, co.IMG_PATIENT)
  draw_pos_size_obj(Perfusionist_pos_size, co.IMG_PERF)
  draw_pos_size_obj(Anesthesiologist_pos_size, co.IMG_ANES)
  draw_pos_size_obj(Cabinet_pos_size, co.IMG_CABINET)
  draw_pos_size_obj(Storage_pos_size, co.IMG_STORAGE)

  Scalpel_stored = game_env["Scalpel_stored"]
  Scalpel_prepared = game_env["Scalpel_prepared"]
  Suture_stored = game_env["Suture_stored"]
  Suture_prepared = game_env["Suture_prepared"]
  Patient_progress = game_env["Patient_progress"]
  CN_pos = game_env["CN_pos"]
  Asked = game_env["Asked"]

  def get_tool_pos_size(s_tool, x_off, y_off):
    tool_w = 0.5
    tool_h = 0.5
    if s_tool == T3SA.ToolLoc.AS:
      return (AS_pos_size[0] + x_off, AS_pos_size[1] + y_off, tool_w, tool_h)
    elif s_tool == T3SA.ToolLoc.SN:
      return (Table_pos_size[0] + x_off, Table_pos_size[1] + y_off, tool_w,
              tool_h)
    elif s_tool == T3SA.ToolLoc.CN:
      return (CN_pos[0] + x_off, CN_pos[1] + y_off, tool_w, tool_h)
    elif s_tool == T3SA.ToolLoc.STORAGE:
      return (Storage_pos_size[0] + x_off, Storage_pos_size[1] + y_off, tool_w,
              tool_h)
    elif s_tool == T3SA.ToolLoc.CABINET:
      return (Cabinet_pos_size[0] + x_off, Cabinet_pos_size[1] + y_off, tool_w,
              tool_h)
    elif s_tool == T3SA.ToolLoc.FLOOR:
      return (6 + x_off, 3 + y_off, tool_w, tool_h)
    else:
      raise NotImplementedError

  draw_pos_size_obj((*CN_pos, 1, 1), co.IMG_CIRCULATING)
  draw_pos_size_obj(get_tool_pos_size(Scalpel_stored, 0.9, 0.1),
                    co.IMG_SCALPEL_STORED)
  draw_pos_size_obj(get_tool_pos_size(Scalpel_prepared, 1.0, 0.1),
                    co.IMG_SCALPEL_PREPARED)
  draw_pos_size_obj(get_tool_pos_size(Suture_stored, 0.9, 0.1),
                    co.IMG_SUTURE_STORED)
  draw_pos_size_obj(get_tool_pos_size(Suture_prepared, 1.0, 0.1),
                    co.IMG_SUTURE_PREPARED)

  return game_objs


def tooldelivery_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[co.DrawingObject], bool] = None) -> List:
  drawing_names = []
  # drawing_names.append("Background")

  for idx, _ in enumerate(game_env["Walls"]):
    img_name = f"Wall{idx}"
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  drawing_names.append(co.IMG_SCRUB)
  drawing_names.append(co.IMG_SURGEON)
  drawing_names.append(co.IMG_PATIENT)
  drawing_names.append(co.IMG_PERF)
  drawing_names.append(co.IMG_ANES)
  drawing_names.append(co.IMG_CABINET)
  drawing_names.append(co.IMG_STORAGE)

  drawing_names.append(co.IMG_SCALPEL_STORED)
  drawing_names.append(co.IMG_SCALPEL_PREPARED)
  drawing_names.append(co.IMG_SUTURE_STORED)
  drawing_names.append(co.IMG_SUTURE_PREPARED)
  drawing_names.append(co.IMG_CIRCULATING)

  return drawing_names


# TOOL HANDOVER HELPERS
def toolhandover_game_scene(
    game_env: Mapping[str, Any],
    game_ltwh: Tuple[int, int, int, int],
    include_background: bool = True,
    cb_is_visible: Callable[[co.DrawingObject], bool] = None
) -> List[co.DrawingObject]:
  game_left, game_top, game_width, game_height = game_ltwh

  # the / 300 and / 200 are hardcoded as 300 x units and 200 y units made up the canvas in the standalone app
  def coord_2_canvas(coord_x, coord_y):
    x = int(game_left + coord_x / 300 * game_width)
    y = int(game_top + coord_y / 200 * game_height)
    return (x, y)

  def size_2_canvas(width, height):
    w = int(width / 300 * game_width)
    h = int(height / 200 * game_height)
    return (w, h)

  # place_w = 0.12
  # place_h = 0.12

  game_objs = []

  # game_objs.append(co.Rectangle("Background", (game_left, game_top), (game_width, game_height), "blue", "green", 0.5))

  def draw_pos_size_obj(pos_size, obj_name):
    x, y, w, h = pos_size
    # the js canvas seems to set input coord as top left corner instead of center like in standalone app, so adjust here
    coord, size = coord_2_canvas(x, y), size_2_canvas(w, h)
    obj = co.GameObject(obj_name,
                        (coord[0] - size[0] / 2, coord[1] - size[1] / 2), size,
                        0, obj_name)
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  icon_sz = 80
  nurse_position = (150, 60, icon_sz, icon_sz)
  surgeon_position = (220, 60, icon_sz, icon_sz)
  patient_position = (205, 90, 200, 150)
  table_position = (60, 110, 120, 180)
  label_text = (270, 0)
  label_text2 = (270, 15)

  tool_sz = 30
  table_part1 = (table_position[0] - 20, table_position[1] - 60, tool_sz,
                 tool_sz)
  table_part2 = (table_position[0] + 20, table_position[1] - 60, tool_sz,
                 tool_sz)
  table_part3 = (table_position[0] - 20, table_position[1] - 20, tool_sz,
                 tool_sz)
  table_part4 = (table_position[0] + 20, table_position[1] - 20, tool_sz,
                 tool_sz)

  tool_types = game_env["tool_types"]
  patient_state = game_env["patient_state"]
  surgeon_sight = game_env["surgeon_sight"]
  surgical_step = game_env["surgical_step"]
  nurse_hand = game_env["nurse_hand"]
  tool_fornow = game_env["tool_fornow"]
  tool_locations = game_env["tool_locations"]
  current_step = game_env["current_step"]

  # background
  draw_pos_size_obj(nurse_position, co.IMG_NURSE)
  draw_pos_size_obj(surgeon_position, co.IMG_SURGEON)
  draw_pos_size_obj(table_position, co.IMG_TABLE)
  draw_pos_size_obj(patient_position, co.IMG_PATIENT)

  # tools
  for idx, ttype in enumerate(tool_types):
    if ttype == tho.Tool_Type.Tool_1:
      image_name = co.IMG_SCALPEL
    elif ttype == tho.Tool_Type.Tool_2:
      image_name = co.IMG_SUTURE
    elif ttype == tho.Tool_Type.Tool_3:
      image_name = co.IMG_FORCEPS
    elif ttype == tho.Tool_Type.Tool_4:
      image_name = co.IMG_SCISSORS

    if tool_locations[idx] == tho.Tool_Location.Nurse:
      img_pos = (nurse_position[0] + 20, nurse_position[1] - 20, tool_sz,
                 tool_sz)
    elif tool_locations[idx] == tho.Tool_Location.Surgeon:
      img_pos = (surgeon_position[0] + 20, surgeon_position[1] - 20, tool_sz,
                 tool_sz)
    elif tool_locations[idx] == tho.Tool_Location.Table_1:
      img_pos = table_part1
    elif tool_locations[idx] == tho.Tool_Location.Table_2:
      img_pos = table_part2
    elif tool_locations[idx] == tho.Tool_Location.Table_3:
      img_pos = table_part3
    elif tool_locations[idx] == tho.Tool_Location.Table_4:
      img_pos = table_part4
    draw_pos_size_obj(img_pos, image_name)

  # surgical step
  x, y = coord_2_canvas(*label_text)
  w = size_2_canvas(75, 0)[0]
  game_objs.append(
      co.TextObject("surgical_step", (x - w / 2, y), w, 25, surgical_step.name))

  # patient state
  x, y = coord_2_canvas(patient_position[0] + 10, patient_position[1] + 20)
  w = size_2_canvas(75, 0)[0]
  game_objs.append(
      co.TextObject("patient_state", (x - w / 2, y), w, 25, patient_state.name))

  # tool for now
  text_tool = "None"
  if tool_fornow is not None:
    text_tool = tool_fornow.name
  x, y = coord_2_canvas(*label_text2)
  w = size_2_canvas(75, 0)[0]
  game_objs.append(co.TextObject("text_tool", (x - w / 2, y), w, 25, text_tool))

  # surgeon sight
  if surgeon_sight == tho.SurgeonSight.Patient:
    game_objs.append(
        co.LineSegment(
            "sight0",
            coord_2_canvas(surgeon_position[0] - 5, surgeon_position[1] - 10),
            coord_2_canvas(patient_position[0], patient_position[1])))
    game_objs.append(
        co.LineSegment(
            "sight1",
            coord_2_canvas(surgeon_position[0] + 5, surgeon_position[1] - 10),
            coord_2_canvas(patient_position[0] + 0.3 * patient_position[2],
                           patient_position[1])))
  else:
    game_objs.append(
        co.LineSegment(
            "sight0",
            coord_2_canvas(surgeon_position[0] - 5, surgeon_position[1] - 15),
            coord_2_canvas(table_position[0] + 0.3 * table_position[2],
                           table_position[1] - 0.4 * table_position[3])))
    game_objs.append(
        co.LineSegment(
            "sight1",
            coord_2_canvas(surgeon_position[0] - 5, surgeon_position[1] - 10),
            coord_2_canvas(table_position[0] + 0.3 * table_position[2],
                           table_position[1])))

  # nurse hand
  arm_st = (nurse_position[0] - 25, nurse_position[1] + 10)
  hand_sz = 5
  if nurse_hand == tho.Tool_Location.Surgeon:
    hand_coord = (surgeon_position[0] - 30, surgeon_position[1])
  elif nurse_hand == tho.Tool_Location.Table_1:
    hand_coord = (table_part1[0] + 15, table_part1[1] - 5)
  elif nurse_hand == tho.Tool_Location.Table_2:
    hand_coord = (table_part2[0] + 15, table_part2[1] - 5)
  elif nurse_hand == tho.Tool_Location.Table_3:
    hand_coord = (table_part3[0] + 15, table_part3[1] - 5)
  elif nurse_hand == tho.Tool_Location.Table_4:
    hand_coord = (table_part4[0] + 15, table_part4[1] - 5)

  mid_coord = (int(0.5 * (hand_coord[0] + arm_st[0])),
               int(0.5 * (hand_coord[1] + arm_st[1])) + 10)

  game_objs.append(
      co.LineSegment("upper arm",
                     coord_2_canvas(*arm_st),
                     coord_2_canvas(*mid_coord),
                     line_color="red",
                     linewidth=3))
  game_objs.append(
      co.LineSegment("lower arm",
                     coord_2_canvas(*mid_coord),
                     coord_2_canvas(*hand_coord),
                     line_color="red",
                     linewidth=3))
  game_objs.append(
      co.Circle("hand", coord_2_canvas(*hand_coord),
                size_2_canvas(hand_sz, hand_sz)[0], "red"))

  game_right = game_left + game_width
  ctrl_btn_w = int(game_width / 12)
  ctrl_btn_w_half = int(game_width / 24)
  x_ctrl_cen = int(game_right + (co.CANVAS_WIDTH - game_right) / 2)
  y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
  game_objs.append(
      co.TextObject("surgeon_buttons", (x_ctrl_cen - int(ctrl_btn_w * 1.5) - 55,
                                        y_ctrl_cen - int(ctrl_btn_w * 3.0)),
                    200, 15, "Surgeon Actions"))
  game_objs.append(
      co.TextObject("nurse_buttons", (x_ctrl_cen + int(ctrl_btn_w * 1.5) - 55,
                                      y_ctrl_cen - int(ctrl_btn_w * 3.0)), 200,
                    15, "Nurse Actions"))

  return game_objs


def toolhandover_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[co.DrawingObject], bool] = None) -> List:
  drawing_names = []
  # drawing_names.append("Background")

  tool_types = game_env["tool_types"]


  # background
  drawing_names.append(co.IMG_NURSE)
  drawing_names.append(co.IMG_SURGEON)
  drawing_names.append(co.IMG_TABLE)
  drawing_names.append(co.IMG_PATIENT)

  # tools
  for idx, ttype in enumerate(tool_types):
    if ttype == tho.Tool_Type.Tool_1:
      image_name = co.IMG_SCALPEL
    elif ttype == tho.Tool_Type.Tool_2:
      image_name = co.IMG_SUTURE
    elif ttype == tho.Tool_Type.Tool_3:
      image_name = co.IMG_FORCEPS
    elif ttype == tho.Tool_Type.Tool_4:
      image_name = co.IMG_SCISSORS
    drawing_names.append(image_name)

  # surgical step
  drawing_names.append("surgical_step")
  # patient state
  drawing_names.append("patient_state")
  # tool for now
  drawing_names.append("text_tool")

  # surgeon sight
  drawing_names.append("sight0")
  drawing_names.append("sight1")

  # nurse hand
  drawing_names.append("upper arm")
  drawing_names.append("lower arm")
  drawing_names.append("hand")

  drawing_names.append("surgeon_buttons")
  drawing_names.append("nurse_buttons")

  return drawing_names


