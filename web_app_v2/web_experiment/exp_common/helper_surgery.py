from typing import Mapping, Any, List, Tuple, Callable, Sequence
import web_experiment.exp_common.canvas_objects as co
import aic_domain.tool_handover_v2.define as tho


# TOOL HANDOVER HELPERS
def toolhandover_game_scene(
    game_env: Mapping[str, Any],
    game_ltwh: Tuple[int, int, int, int],
    include_background: bool = True,
    cb_is_visible: Callable[[co.DrawingObject], bool] = None
) -> List[co.DrawingObject]:
  game_left, game_top, game_width, game_height = game_ltwh

  grid_w = game_env["width"]
  grid_h = game_env["height"]

  def coord_2_canvas(coord_x, coord_y):
    x = int(game_left + coord_x / grid_w * game_width)
    y = int(game_top + coord_y / grid_h * game_height)
    return (x, y)

  def size_2_canvas(width, height):
    w = int(width / grid_w * game_width)
    h = int(height / grid_h * game_height)
    return (w, h)

  game_objs = []

  def draw_pos_size_obj(pos_size, obj_name):
    x, y, w, h = pos_size
    # the js canvas seems to set input coord as top left corner instead of center like in standalone app, so adjust here
    coord, size = coord_2_canvas(x, y), size_2_canvas(w, h)
    obj = co.GameObject(obj_name,
                        (coord[0] - size[0] / 2, coord[1] - size[1] / 2), size,
                        0, obj_name)
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  rad = size_2_canvas(0.45, 1)[0]

  surgeon_pos = game_env["surgeon_pos"]
  obj = co.Circle(co.IMG_SURGEON,
                  coord_2_canvas(surgeon_pos[0] + 0.5, surgeon_pos[1] + 0.5),
                  rad, "cyan")
  game_objs.append(obj)

  perf_pos = game_env["perf_pos"]
  obj = co.Circle("Perf", coord_2_canvas(perf_pos[0] + 0.5, perf_pos[1] + 0.5),
                  rad, "gray")
  game_objs.append(obj)

  anes_pos = game_env["anes_pos"]
  obj = co.Circle("Anes", coord_2_canvas(anes_pos[0] + 0.5, anes_pos[1] + 0.5),
                  rad, "black")
  game_objs.append(obj)

  pat_pos_size = game_env["patient_pos_size"]
  obj = co.Rectangle(co.IMG_PATIENT,
                     coord_2_canvas(pat_pos_size[0], pat_pos_size[1]),
                     size_2_canvas(pat_pos_size[2], pat_pos_size[3]), "blue")
  game_objs.append(obj)

  table_blocks = game_env["table_blocks"]
  for idx, table in enumerate(table_blocks):
    obj = co.Rectangle(co.IMG_TABLE + str(idx),
                       coord_2_canvas(table[0], table[1]), size_2_canvas(1, 1),
                       "green")
    game_objs.append(obj)

  vital_pos = game_env["vital_pos"]
  pat_vital = game_env["patient_vital"].name
  obj = co.TextObject("vital", coord_2_canvas(vital_pos[0], vital_pos[1]),
                      size_2_canvas(1, 0)[0], 15, pat_vital)
  game_objs.append(obj)

  nurse_pos = game_env["nurse_pos"]
  nurse_dir = game_env["nurse_dir"]
  obj = co.Circle(co.IMG_NURSE,
                  coord_2_canvas(nurse_pos[0] + 0.5, nurse_pos[1] + 0.5), rad,
                  "red")
  game_objs.append(obj)

  if nurse_dir == tho.NurseDirection.Up:
    obj = co.Rectangle("nurse_dir",
                       coord_2_canvas(nurse_pos[0] + 0.4, nurse_pos[1]),
                       size_2_canvas(0.2, 0.2), "red")
  elif nurse_dir == tho.NurseDirection.Left:
    obj = co.Rectangle("nurse_dir",
                       coord_2_canvas(nurse_pos[0], nurse_pos[1] + 0.4),
                       size_2_canvas(0.2, 0.2), "red")
  elif nurse_dir == tho.NurseDirection.Down:
    obj = co.Rectangle("nurse_dir",
                       coord_2_canvas(nurse_pos[0] + 0.4, nurse_pos[1] + 0.8),
                       size_2_canvas(0.2, 0.2), "red")
  elif nurse_dir == tho.NurseDirection.Right:
    obj = co.Rectangle("nurse_dir",
                       coord_2_canvas(nurse_pos[0] + 0.8, nurse_pos[1] + 0.4),
                       size_2_canvas(0.2, 0.2), "red")
  game_objs.append(obj)

  # if data["surgeon_ready"]:
  #   self.create_text((surgeon_pos[0] + 0.5) * x_unit,
  #                    (surgeon_pos[1] + 0.1) * y_unit, "Ready")
  # if data["anes_ready"]:
  #   self.create_text((anes_pos[0] + 0.5) * x_unit,
  #                    (anes_pos[1] + 0.1) * y_unit, "Ready")
  # if data["perf_ready"]:
  #   self.create_text((perf_pos[0] + 0.5) * x_unit,
  #                    (perf_pos[1] + 0.1) * y_unit, "Ready")

  nurse_tool = game_env["nurse_tool"]
  surgeon_tool = game_env["surgeon_tool"]
  tool_table_zone = game_env["tool_table_zone"]
  for tool, tb_zone in tool_table_zone.items():
    tool_pos = None
    if tool == nurse_tool:
      tool_pos = (nurse_pos[0] + 0.35, nurse_pos[1] + 0.35)
    elif tool == surgeon_tool:
      tool_pos = (surgeon_pos[0] + 0.35, surgeon_pos[1] + 0.35)
    else:
      table_lt = table_blocks[tb_zone[0]]
      if tb_zone[1] == 1:
        tool_pos = (table_lt[0] + 0.65, table_lt[1] + 0.15)
      elif tb_zone[1] == 2:
        tool_pos = (table_lt[0] + 0.15, table_lt[1] + 0.15)
      elif tb_zone[1] == 3:
        tool_pos = (table_lt[0] + 0.15, table_lt[1] + 0.65)
      elif tb_zone[1] == 4:
        tool_pos = (table_lt[0] + 0.65, table_lt[1] + 0.65)
    obj = co.Rectangle(tool.name, coord_2_canvas(*tool_pos),
                       size_2_canvas(0.3, 0.3), "yellow")
    game_objs.append(obj)
    obj = co.TextObject(tool.name + "label", coord_2_canvas(*tool_pos),
                        size_2_canvas(0.4, 0)[0], 15, tool.name[:2])

    game_objs.append(obj)

  visible_game_objs = []
  for obj in game_objs:
    if cb_is_visible is None or cb_is_visible(obj):
      visible_game_objs.append(obj)

  return game_objs


def toolhandover_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[co.DrawingObject], bool] = None) -> List:
  drawing_names = []
  # drawing_names.append("Background")

  drawing_names.append(co.IMG_SURGEON)
  drawing_names.append("Perf")
  drawing_names.append("Anes")
  drawing_names.append(co.IMG_PATIENT)

  table_blocks = game_env["table_blocks"]
  for idx, table in enumerate(table_blocks):
    drawing_names.append(co.IMG_TABLE + str(idx))

  drawing_names.append("vital")
  drawing_names.append(co.IMG_NURSE)
  drawing_names.append("nurse_dir")

  tool_table_zone = game_env["tool_table_zone"]
  for tool, tb_zone in tool_table_zone.items():
    drawing_names.append(tool.name)
    drawing_names.append(tool.name + "label")

  return drawing_names
