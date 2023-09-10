from dataclasses import dataclass
from typing import Mapping, Any, List, Tuple, Callable, Sequence
import os
import time
import numpy as np
from aic_domain.box_push import conv_box_idx_2_state, BoxState
from aic_domain.rescue import (Place, Route, Location, E_Type, PlaceName, Work,
                               is_work_done)
import aic_domain.tooldelivery.tooldelivery_v3_state_action as T3SA
import aic_domain.tool_handover.define as tho
import web_experiment.exp_common.canvas_objects as co


@dataclass
class DrawInfo:
  img_name: str
  offset: Tuple[float, float]
  size: Tuple[float, float]
  angle: float
  circles: Sequence[Tuple[float, float, float]]


RESCUE_PLACE_DRAW_INFO = {
    PlaceName.Fire_stateion:
    DrawInfo(co.IMG_FIRE_STATION, (0.05, -0.1), (0.12, 0.12),
             0,
             circles=[(0.06, -0.12, 0.1), (0, 0, 0.06)]),
    PlaceName.Police_station:
    DrawInfo(co.IMG_POLICE_STATION, (-0.09, 0), (0.12, 0.12),
             0,
             circles=[(-0.15, 0.03, 0.16), (0, 0, 0.06)]),
    PlaceName.Campsite:
    DrawInfo(co.IMG_CAMPSITE, (0, -0.07), (0.155, 0.12),
             0,
             circles=[(0.01, -0.17, 0.2)]),
    PlaceName.City_hall:
    DrawInfo(co.IMG_CITY_HALL, (-0.03, -0.05), (0.12, 0.1),
             0,
             circles=[(-0.09, -0.1, 0.15), (0, 0, 0.06)]),
    PlaceName.Mall:
    DrawInfo(co.IMG_MALL, (0, -0.05), (0.145, 0.145),
             0,
             circles=[(0.02, 0, 0.33)]),
    PlaceName.Bridge_1:
    DrawInfo(co.IMG_BRIDGE, (0.06, 0.02), (0.12, 0.03),
             0.15 * np.pi,
             circles=[(0, 0, 0.04)]),
    PlaceName.Bridge_2:
    DrawInfo(co.IMG_BRIDGE, (0.06, 0.02), (0.12, 0.03),
             0.15 * np.pi,
             circles=[(0, 0, 0.04)]),
}


def store_user_label_locally(user_label_path, user_id, session_name,
                             user_labels):
  file_name = get_user_label_file_name(user_label_path, user_id, session_name)
  dir_path = os.path.dirname(file_name)
  if dir_path != '' and not os.path.exists(dir_path):
    os.makedirs(dir_path)

  with open(file_name, 'w', newline='') as txtfile:
    # sequence
    txtfile.write('# cur_step, user_label\n')

    for tup_label in user_labels:
      txtfile.write('%d; %s;' % tup_label)
      txtfile.write('\n')


def get_user_label_file_name(user_label_path, user_id, session_name):
  traj_dir = os.path.join(user_label_path, user_id)

  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = ('user_label_' + session_name + '_' + str(user_id) + '_' +
               time_stamp + '.txt')
  return os.path.join(traj_dir, file_name)


def get_file_name(save_path, user_id, session_name):
  traj_dir = os.path.join(save_path, user_id)
  # save somewhere
  if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

  sec, msec = divmod(time.time() * 1000, 1000)
  time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                          time.gmtime(sec)), msec)
  file_name = session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
  return os.path.join(traj_dir, file_name)


def get_btn_boxpush_actions(game_width,
                            game_right,
                            up_disable=False,
                            down_disable=False,
                            left_disable=False,
                            right_disable=False,
                            stay_disable=False,
                            pickup_disable=False,
                            drop_disable=False,
                            select_disable=True):
  ctrl_btn_w = int(game_width / 12)
  ctrl_btn_w_half = int(game_width / 24)
  x_ctrl_cen = int(game_right + (co.CANVAS_WIDTH - game_right) / 2)
  y_ctrl_cen = int(co.CANVAS_HEIGHT * 0.65)
  x_joy_cen = int(x_ctrl_cen - ctrl_btn_w * 1.5)
  btn_stay = co.JoystickStay((x_joy_cen, y_ctrl_cen),
                             ctrl_btn_w,
                             disable=stay_disable)
  btn_up = co.JoystickUp((x_joy_cen, y_ctrl_cen - ctrl_btn_w_half),
                         ctrl_btn_w,
                         disable=up_disable)
  btn_right = co.JoystickRight((x_joy_cen + ctrl_btn_w_half, y_ctrl_cen),
                               ctrl_btn_w,
                               disable=right_disable)
  btn_down = co.JoystickDown((x_joy_cen, y_ctrl_cen + ctrl_btn_w_half),
                             ctrl_btn_w,
                             disable=down_disable)
  btn_left = co.JoystickLeft((x_joy_cen - ctrl_btn_w_half, y_ctrl_cen),
                             ctrl_btn_w,
                             disable=left_disable)
  font_size = 20
  btn_pickup = co.ButtonRect(
      co.BTN_PICKUP,
      (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen - int(ctrl_btn_w * 0.6)),
      (ctrl_btn_w * 2, ctrl_btn_w),
      font_size,
      "Pick Up",
      disable=pickup_disable)
  btn_drop = co.ButtonRect(
      co.BTN_DROP,
      (x_ctrl_cen + int(ctrl_btn_w * 1.5), y_ctrl_cen + int(ctrl_btn_w * 0.6)),
      (ctrl_btn_w * 2, ctrl_btn_w),
      font_size,
      "Drop",
      disable=drop_disable)
  btn_select = co.ButtonRect(co.BTN_SELECT,
                             (x_ctrl_cen, y_ctrl_cen + ctrl_btn_w * 2),
                             (ctrl_btn_w * 4, ctrl_btn_w),
                             font_size,
                             "Select Destination",
                             disable=select_disable)
  return (btn_up, btn_down, btn_left, btn_right, btn_stay, btn_pickup, btn_drop,
          btn_select)


def boxpush_game_scene(
    game_env: Mapping[str, Any],
    game_lwth: Tuple[int, int, int, int],
    is_movers: bool,
    include_background: bool = True,
    cb_is_visible: Callable[[co.DrawingObject], bool] = None
) -> List[co.DrawingObject]:
  x_grid = game_env["x_grid"]
  y_grid = game_env["y_grid"]

  game_left, game_top, game_width, game_height = game_lwth

  def coord_2_canvas(coord_x, coord_y):
    x = int(game_left + coord_x / x_grid * game_width)
    y = int(game_top + coord_y / y_grid * game_height)
    return (x, y)

  def size_2_canvas(width, height):
    w = int(width / x_grid * game_width)
    h = int(height / y_grid * game_height)
    return (w, h)

  game_objs = []
  if include_background:
    for idx, coord in enumerate(game_env["boxes"]):
      game_pos = coord_2_canvas(coord[0] + 0.5, coord[1] + 0.7)
      size = size_2_canvas(0.4, 0.2)
      obj = co.Ellipse(co.BOX_ORIGIN + str(idx), game_pos, size, "grey")
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    for idx, coord in enumerate(game_env["walls"]):
      wid = 1
      hei = 1
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      angle = 0 if game_env["wall_dir"][idx] == 0 else 0.5 * np.pi
      obj = co.GameObject(co.IMG_WALL + str(idx), coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), angle, co.IMG_WALL)
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    for idx, coord in enumerate(game_env["goals"]):
      hei = 1
      wid = 1
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = co.GameObject(co.IMG_GOAL + str(idx), coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, co.IMG_GOAL)
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

  num_drops = len(game_env["drops"])
  num_goals = len(game_env["goals"])
  a1_hold_box = -1
  a2_hold_box = -1
  for idx, bidx in enumerate(game_env["box_states"]):
    state = conv_box_idx_2_state(bidx, num_drops, num_goals)
    obj = None
    if state[0] == BoxState.Original:
      coord = game_env["boxes"][idx]
      wid = 0.541
      hei = 0.60
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      img_name = co.IMG_BOX if is_movers else co.IMG_TRASH_BAG
      obj = co.GameObject(img_name + str(idx), coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, img_name)
    elif state[0] == BoxState.WithAgent1:  # with a1
      coord = game_env["a1_pos"]
      offset = 0
      if game_env["a2_pos"] == coord:
        offset = -0.2
      hei = 1
      wid = 0.385
      left = coord[0] + 0.5 + offset - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = co.GameObject(co.IMG_MAN_BAG, coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, co.IMG_MAN_BAG)
      a1_hold_box = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      coord = game_env["a2_pos"]
      offset = 0
      if game_env["a1_pos"] == coord:
        offset = -0.2
      hei = 0.8
      wid = 0.476
      left = coord[0] + 0.5 + offset - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = co.GameObject(co.IMG_ROBOT_BAG, coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, co.IMG_ROBOT_BAG)
      a2_hold_box = idx
    elif state[0] == BoxState.WithBoth:  # with both
      coord = game_env["a1_pos"]
      hei = 1
      wid = 0.712
      left = coord[0] + 0.5 - 0.5 * wid
      top = coord[1] + 0.5 - 0.5 * hei
      obj = co.GameObject(co.IMG_BOTH_BOX, coord_2_canvas(left, top),
                          size_2_canvas(wid, hei), 0, co.IMG_BOTH_BOX)
      a1_hold_box = idx
      a2_hold_box = idx

    if obj is not None:
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

  if a1_hold_box < 0:
    coord = game_env["a1_pos"]
    offset = 0
    if coord == game_env["a2_pos"]:
      offset = 0.2 if a2_hold_box >= 0 else -0.2
    hei = 1
    wid = 0.23
    left = coord[0] + 0.5 + offset - 0.5 * wid
    top = coord[1] + 0.5 - 0.5 * hei
    img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
    obj = co.GameObject(img_name, coord_2_canvas(left, top),
                        size_2_canvas(wid, hei), 0, img_name)
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  if a2_hold_box < 0:
    coord = game_env["a2_pos"]
    offset = 0
    if coord == game_env["a1_pos"]:
      offset = 0.2
    hei = 0.8
    wid = 0.422
    left = coord[0] + 0.5 + offset - 0.5 * wid
    top = coord[1] + 0.5 - 0.5 * hei
    obj = co.GameObject(co.IMG_ROBOT, coord_2_canvas(left, top),
                        size_2_canvas(wid, hei), 0, co.IMG_ROBOT)
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  return game_objs


def boxpush_game_scene_names(
    game_env: Mapping[str, Any],
    is_movers: bool,
    cb_is_visible: Callable[[str], bool] = None) -> List:

  drawing_names = []
  for idx, _ in enumerate(game_env["boxes"]):
    img_name = co.BOX_ORIGIN + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  for idx, _ in enumerate(game_env["walls"]):
    img_name = co.IMG_WALL + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  for idx, _ in enumerate(game_env["goals"]):
    img_name = co.IMG_GOAL + str(idx)
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  num_drops = len(game_env["drops"])
  num_goals = len(game_env["goals"])
  a1_hold_box = -1
  a2_hold_box = -1
  for idx, bidx in enumerate(game_env["box_states"]):
    state = conv_box_idx_2_state(bidx, num_drops, num_goals)
    img_name = None
    if state[0] == BoxState.Original:
      img_type = co.IMG_BOX if is_movers else co.IMG_TRASH_BAG
      img_name = img_type + str(idx)
    elif state[0] == BoxState.WithAgent1:  # with a1
      img_name = co.IMG_MAN_BAG
      a1_hold_box = idx
    elif state[0] == BoxState.WithAgent2:  # with a2
      img_name = co.IMG_ROBOT_BAG
      a2_hold_box = idx
    elif state[0] == BoxState.WithBoth:  # with both
      img_name = co.IMG_BOTH_BOX
      a1_hold_box = idx
      a2_hold_box = idx

    if img_name is not None:
      if cb_is_visible is None or cb_is_visible(img_name):
        drawing_names.append(img_name)

  if a1_hold_box < 0:
    img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  if a2_hold_box < 0:
    img_name = co.IMG_ROBOT
    if cb_is_visible is None or cb_is_visible(img_name):
      drawing_names.append(img_name)

  return drawing_names


def location_2_coord(loc: Location, places: Sequence[Place],
                     routes: Sequence[Route]):
  if loc.type == E_Type.Place:
    return places[loc.id].coord
  else:
    route_id = loc.id  # type: int
    route = routes[route_id]  # type: Route
    idx = loc.index
    return route.coords[idx]


################################################################################
# NOTE: test codes by Arnav
TEST_ARNAV = True
if TEST_ARNAV:

  def rescue_game_scene(
      game_env: Mapping[str, Any],
      game_lwth: Tuple[int, int, int, int],
      include_background: bool = True,
      cb_is_visible: Callable[[co.DrawingObject], bool] = None
  ) -> List[co.DrawingObject]:
    game_left, game_top, game_width, game_height = game_lwth

    def coord_2_canvas(coord_x, coord_y):
      x = int(game_left + coord_x * game_width)
      y = int(game_top + coord_y * game_height)
      return (x, y)

    def size_2_canvas(width, height):
      w = int(width * game_width)
      h = int(height * game_height)
      return (w, h)

    # place_w = 0.12
    # place_h = 0.12

    places = game_env["places"]  # type: Sequence[Place]
    routes = game_env["routes"]  # type: Sequence[Route]
    game_objs = []

    def add_obj(obj):
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    obj = co.Circle("hi", (100, 100), 30, fill_color="green", alpha=0.8)
    add_obj(obj)

    return game_objs

  def rescue_game_scene_names(
      game_env: Mapping[str, Any],
      cb_is_visible: Callable[[str], bool] = None) -> List:

    drawing_names = []

    def add_obj_name(obj_name):
      if cb_is_visible is None or cb_is_visible(obj_name):
        drawing_names.append(obj_name)

    add_obj_name("hi")

    return drawing_names
else:

  def rescue_game_scene(
      game_env: Mapping[str, Any],
      game_lwth: Tuple[int, int, int, int],
      include_background: bool = True,
      cb_is_visible: Callable[[co.DrawingObject], bool] = None
  ) -> List[co.DrawingObject]:
    game_left, game_top, game_width, game_height = game_lwth

    def coord_2_canvas(coord_x, coord_y):
      x = int(game_left + coord_x * game_width)
      y = int(game_top + coord_y * game_height)
      return (x, y)

    def size_2_canvas(width, height):
      w = int(width * game_width)
      h = int(height * game_height)
      return (w, h)

    # place_w = 0.12
    # place_h = 0.12

    places = game_env["places"]  # type: Sequence[Place]
    routes = game_env["routes"]  # type: Sequence[Route]
    game_objs = []

    def add_obj(obj):
      if cb_is_visible is None or cb_is_visible(obj):
        game_objs.append(obj)

    font_size = 15
    if include_background:
      river_coords = [
          coord_2_canvas(0.68, 1),
          coord_2_canvas(0.75, 0.9),
          coord_2_canvas(0.77, 0.83),
          coord_2_canvas(0.85, 0.77),
          coord_2_canvas(1, 0.7)
      ]
      obj = co.Curve(co.IMG_BACKGROUND, river_coords, 30, "blue")
      add_obj(obj)

      for idx, route in enumerate(routes):
        list_coord = []
        list_coord.append(coord_2_canvas(*places[route.start].coord))
        for coord in route.coords:
          list_coord.append(coord_2_canvas(*coord))
        list_coord.append(coord_2_canvas(*places[route.end].coord))

        obj = co.Curve(co.IMG_ROUTE + str(idx), list_coord, 10, "grey")
        add_obj(obj)

      def add_place(place: Place, offset, size, img_name):
        name = place.name
        building_pos = np.array(place.coord) + np.array(offset)
        canvas_pt = coord_2_canvas(*place.coord)

        wid = size[0]
        hei = size[1]
        size_cnvs = size_2_canvas(wid, hei)
        game_pos = coord_2_canvas(building_pos[0] - wid / 2,
                                  building_pos[1] - hei / 2)
        text_width = size_cnvs[0] * 2
        text_pos = (int(game_pos[0] + 0.5 * size_cnvs[0] - 0.5 * text_width),
                    int(game_pos[1] - font_size))
        obj = co.Circle("ground" + name, canvas_pt,
                        size_2_canvas(0.03, 0)[0], "grey")
        add_obj(obj)
        obj = co.GameObject(name, game_pos, size_cnvs, 0, img_name)
        add_obj(obj)
        obj = co.TextObject("text" + name, text_pos, text_width, font_size,
                            name, "center")
        add_obj(obj)

        p_wid = 0.025
        p_hei = p_wid * 2.5
        p_size_cnvs = size_2_canvas(p_wid, p_hei)
        for pidx in range(place.helps):
          if pidx < 2:
            p_x = game_pos[0] - p_size_cnvs[0] * (pidx + 1)
            p_y = game_pos[1] + size_cnvs[1] - p_size_cnvs[1]
          else:
            p_x = game_pos[0] - p_size_cnvs[0] * (pidx - 1)
            p_y = game_pos[1] + size_cnvs[1] - 2 * p_size_cnvs[1]
          obj = co.GameObject("human" + name + str(pidx), (p_x, p_y),
                              p_size_cnvs, 0, co.IMG_HUMAN)
          add_obj(obj)

      for idx in [0, 1, 2, 4, 5]:
        place_name = places[idx].name
        offset = RESCUE_PLACE_DRAW_INFO[place_name].offset
        size = RESCUE_PLACE_DRAW_INFO[place_name].size
        img_name = RESCUE_PLACE_DRAW_INFO[place_name].img_name
        add_place(places[idx], offset, size, img_name)

    work_locations = game_env["work_locations"]  # type: Sequence[Location]
    work_states = game_env["work_states"]
    work_info = game_env["work_info"]  # type: Sequence[Work]

    pos_a1 = location_2_coord(game_env["a1_pos"], places, routes)
    pos_a2 = location_2_coord(game_env["a2_pos"], places, routes)
    wid_a = 0.085
    hei_a = 0.085
    offset_x_a1 = 0
    offset_y_a1 = 0
    offset_x_a2 = 0
    offset_y_a2 = 0
    for idx, wstate in enumerate(work_states):
      if wstate != 0:
        loc = work_locations[idx]
        pos = location_2_coord(loc, places, routes)
        if pos == pos_a1:
          offset_x_a1 = -wid_a * 0.7 / 2
          offset_y_a1 = hei_a * 0.5 / 2
        if pos == pos_a2:
          offset_x_a2 = wid_a * 0.7 / 2
          offset_y_a2 = -hei_a * 0.5 / 2

        wid = 0.06
        hei = 0.06
        offset_x = 0
        offset_y = 0
        game_pos = coord_2_canvas(pos[0] + offset_x - wid / 2,
                                  pos[1] + offset_y - hei / 2)
        size_cnvs = size_2_canvas(wid, hei)
        obj = co.GameObject(co.IMG_WORK + str(idx), game_pos, size_cnvs, 0,
                            co.IMG_WORK)
        add_obj(obj)
      else:
        if work_locations[idx].type == E_Type.Place:
          place = places[work_locations[idx].id]
          if place.name in [PlaceName.Bridge_1, PlaceName.Bridge_2]:
            offset = RESCUE_PLACE_DRAW_INFO[place.name].offset
            size = RESCUE_PLACE_DRAW_INFO[place.name].size
            angle = RESCUE_PLACE_DRAW_INFO[place.name].angle
            img_name = RESCUE_PLACE_DRAW_INFO[place.name].img_name
            building_pos = np.array(place.coord) + np.array(offset)
            wid, hei = size
            size_cnvs = size_2_canvas(wid, hei)
            game_pos = coord_2_canvas(building_pos[0] - wid / 2,
                                      building_pos[1] - hei / 2)
            obj = co.GameObject(place.name, game_pos, size_cnvs, angle,
                                img_name)
            add_obj(obj)

    if pos_a1 == pos_a2 and offset_x_a1 == 0:
      offset_x_a1 = -wid_a * 0.7 / 2
      offset_y_a1 = hei_a * 0.5 / 2
      offset_x_a2 = wid_a * 0.7 / 2
      offset_y_a2 = -hei_a * 0.5 / 2

    game_pos_a1 = coord_2_canvas(pos_a1[0] + offset_x_a1 - wid_a / 2,
                                 pos_a1[1] + offset_y_a1 - hei_a / 2)
    size_a1 = size_2_canvas(wid_a, hei_a)

    game_pos_a2 = coord_2_canvas(pos_a2[0] + offset_x_a2 - wid_a / 2,
                                 pos_a2[1] + offset_y_a2 - hei_a / 2)
    size_a2 = size_2_canvas(wid_a, hei_a)

    obj = co.GameObject(co.IMG_POLICE_CAR, game_pos_a1, size_a1, 0,
                        co.IMG_POLICE_CAR)
    add_obj(obj)
    obj = co.GameObject(co.IMG_FIRE_ENGINE, game_pos_a2, size_a2, 0,
                        co.IMG_FIRE_ENGINE)
    add_obj(obj)

    return game_objs

  def rescue_game_scene_names(
      game_env: Mapping[str, Any],
      cb_is_visible: Callable[[str], bool] = None) -> List:

    drawing_names = []

    def add_obj_name(obj_name):
      if cb_is_visible is None or cb_is_visible(obj_name):
        drawing_names.append(obj_name)

    add_obj_name(co.IMG_BACKGROUND)

    routes = game_env["routes"]  # type: Sequence[Route]
    for idx, _ in enumerate(routes):
      add_obj_name(co.IMG_ROUTE + str(idx))

    work_locations = game_env["work_locations"]  # type: Sequence[Location]
    work_states = game_env["work_states"]
    places = game_env["places"]  # type: Sequence[Place]
    for idx, wstate in enumerate(work_states):
      if wstate == 0:
        if work_locations[idx].type == E_Type.Place:
          place = places[work_locations[idx].id]
          if place.name in [PlaceName.Bridge_1, PlaceName.Bridge_2]:
            add_obj_name(place.name)

    for idx in [0, 1, 2, 4, 5]:
      add_obj_name("ground" + places[idx].name)
      add_obj_name(places[idx].name)
      add_obj_name("text" + places[idx].name)

    for idx, wstate in enumerate(work_states):
      if wstate != 0:
        add_obj_name(co.IMG_WORK + str(idx))

    work_info = game_env["work_info"]  # type: Sequence[Work]
    for idx, _ in enumerate(work_states):
      if not is_work_done(idx, work_states, work_info[idx].coupled_works):
        place = places[work_info[idx].rescue_place]
        num_help = place.helps
        for pidx in range(num_help):
          add_obj_name("human" + place.name + str(pidx))

    add_obj_name(co.IMG_POLICE_CAR)
    add_obj_name(co.IMG_FIRE_ENGINE)

    return drawing_names


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


# game_objs = []
#   if include_background:
#     for idx, coord in enumerate(game_env["boxes"]):
#       game_pos = coord_2_canvas(coord[0] + 0.5, coord[1] + 0.7)
#       size = size_2_canvas(0.4, 0.2)
#       obj = co.Ellipse(co.BOX_ORIGIN + str(idx), game_pos, size, "grey")
#       if cb_is_visible is None or cb_is_visible(obj):
#         game_objs.append(obj)

#     for idx, coord in enumerate(game_env["walls"]):
#       wid = 1
#       hei = 1
#       left = coord[0] + 0.5 - 0.5 * wid
#       top = coord[1] + 0.5 - 0.5 * hei
#       angle = 0 if game_env["wall_dir"][idx] == 0 else 0.5 * np.pi
#       obj = co.GameObject(co.IMG_WALL + str(idx), coord_2_canvas(left, top),
#                           size_2_canvas(wid, hei), angle, co.IMG_WALL)
#       if cb_is_visible is None or cb_is_visible(obj):
#         game_objs.append(obj)

#     for idx, coord in enumerate(game_env["goals"]):
#       hei = 1
#       wid = 1
#       left = coord[0] + 0.5 - 0.5 * wid
#       top = coord[1] + 0.5 - 0.5 * hei
#       obj = co.GameObject(co.IMG_GOAL + str(idx), coord_2_canvas(left, top),
#                           size_2_canvas(wid, hei), 0, co.IMG_GOAL)
#       if cb_is_visible is None or cb_is_visible(obj):
#         game_objs.append(obj)

#   num_drops = len(game_env["drops"])
#   num_goals = len(game_env["goals"])
#   a1_hold_box = -1
#   a2_hold_box = -1
#   for idx, bidx in enumerate(game_env["box_states"]):
#     state = conv_box_idx_2_state(bidx, num_drops, num_goals)
#     obj = None
#     if state[0] == BoxState.Original:
#       coord = game_env["boxes"][idx]
#       wid = 0.541
#       hei = 0.60
#       left = coord[0] + 0.5 - 0.5 * wid
#       top = coord[1] + 0.5 - 0.5 * hei
#       img_name = co.IMG_BOX if is_movers else co.IMG_TRASH_BAG
#       obj = co.GameObject(img_name + str(idx), coord_2_canvas(left, top),
#                           size_2_canvas(wid, hei), 0, img_name)
#     elif state[0] == BoxState.WithAgent1:  # with a1
#       coord = game_env["a1_pos"]
#       offset = 0
#       if game_env["a2_pos"] == coord:
#         offset = -0.2
#       hei = 1
#       wid = 0.385
#       left = coord[0] + 0.5 + offset - 0.5 * wid
#       top = coord[1] + 0.5 - 0.5 * hei
#       obj = co.GameObject(co.IMG_MAN_BAG, coord_2_canvas(left, top),
#                           size_2_canvas(wid, hei), 0, co.IMG_MAN_BAG)
#       a1_hold_box = idx
#     elif state[0] == BoxState.WithAgent2:  # with a2
#       coord = game_env["a2_pos"]
#       offset = 0
#       if game_env["a1_pos"] == coord:
#         offset = -0.2
#       hei = 0.8
#       wid = 0.476
#       left = coord[0] + 0.5 + offset - 0.5 * wid
#       top = coord[1] + 0.5 - 0.5 * hei
#       obj = co.GameObject(co.IMG_ROBOT_BAG, coord_2_canvas(left, top),
#                           size_2_canvas(wid, hei), 0, co.IMG_ROBOT_BAG)
#       a2_hold_box = idx
#     elif state[0] == BoxState.WithBoth:  # with both
#       coord = game_env["a1_pos"]
#       hei = 1
#       wid = 0.712
#       left = coord[0] + 0.5 - 0.5 * wid
#       top = coord[1] + 0.5 - 0.5 * hei
#       obj = co.GameObject(co.IMG_BOTH_BOX, coord_2_canvas(left, top),
#                           size_2_canvas(wid, hei), 0, co.IMG_BOTH_BOX)
#       a1_hold_box = idx
#       a2_hold_box = idx

#     if obj is not None:
#       if cb_is_visible is None or cb_is_visible(obj):
#         game_objs.append(obj)

#   if a1_hold_box < 0:
#     coord = game_env["a1_pos"]
#     offset = 0
#     if coord == game_env["a2_pos"]:
#       offset = 0.2 if a2_hold_box >= 0 else -0.2
#     hei = 1
#     wid = 0.23
#     left = coord[0] + 0.5 + offset - 0.5 * wid
#     top = coord[1] + 0.5 - 0.5 * hei
#     img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
#     obj = co.GameObject(img_name, coord_2_canvas(left, top),
#                         size_2_canvas(wid, hei), 0, img_name)
#     if cb_is_visible is None or cb_is_visible(obj):
#       game_objs.append(obj)

#   if a2_hold_box < 0:
#     coord = game_env["a2_pos"]
#     offset = 0
#     if coord == game_env["a1_pos"]:
#       offset = 0.2
#     hei = 0.8
#     wid = 0.422
#     left = coord[0] + 0.5 + offset - 0.5 * wid
#     top = coord[1] + 0.5 - 0.5 * hei
#     obj = co.GameObject(co.IMG_ROBOT, coord_2_canvas(left, top),
#                         size_2_canvas(wid, hei), 0, co.IMG_ROBOT)
#     if cb_is_visible is None or cb_is_visible(obj):
#       game_objs.append(obj)

#   return game_objs


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


# def boxpush_game_scene_names(
#         game_env: Mapping[str, Any],
#         is_movers: bool,
#         cb_is_visible: Callable[[str], bool] = None) -> List:

#   drawing_names = []
#   for idx, _ in enumerate(game_env["boxes"]):
#     img_name = co.BOX_ORIGIN + str(idx)
#     if cb_is_visible is None or cb_is_visible(img_name):
#       drawing_names.append(img_name)

#   for idx, _ in enumerate(game_env["walls"]):
#     img_name = co.IMG_WALL + str(idx)
#     if cb_is_visible is None or cb_is_visible(img_name):
#       drawing_names.append(img_name)

#   for idx, _ in enumerate(game_env["goals"]):
#     img_name = co.IMG_GOAL + str(idx)
#     if cb_is_visible is None or cb_is_visible(img_name):
#       drawing_names.append(img_name)

#   num_drops = len(game_env["drops"])
#   num_goals = len(game_env["goals"])
#   a1_hold_box = -1
#   a2_hold_box = -1
#   for idx, bidx in enumerate(game_env["box_states"]):
#     state = conv_box_idx_2_state(bidx, num_drops, num_goals)
#     img_name = None
#     if state[0] == BoxState.Original:
#       img_type = co.IMG_BOX if is_movers else co.IMG_TRASH_BAG
#       img_name = img_type + str(idx)
#     elif state[0] == BoxState.WithAgent1:  # with a1
#       img_name = co.IMG_MAN_BAG
#       a1_hold_box = idx
#     elif state[0] == BoxState.WithAgent2:  # with a2
#       img_name = co.IMG_ROBOT_BAG
#       a2_hold_box = idx
#     elif state[0] == BoxState.WithBoth:  # with both
#       img_name = co.IMG_BOTH_BOX
#       a1_hold_box = idx
#       a2_hold_box = idx

#     if img_name is not None:
#       if cb_is_visible is None or cb_is_visible(img_name):
#         drawing_names.append(img_name)

#   if a1_hold_box < 0:
#     img_name = co.IMG_WOMAN if is_movers else co.IMG_MAN
#     if cb_is_visible is None or cb_is_visible(img_name):
#       drawing_names.append(img_name)

#   if a2_hold_box < 0:
#     img_name = co.IMG_ROBOT
#     if cb_is_visible is None or cb_is_visible(img_name):
#       drawing_names.append(img_name)

#   return drawing_names


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
  patient_state = game_env["patient_state"]
  surgeon_sight = game_env["surgeon_sight"]
  surgical_step = game_env["surgical_step"]
  nurse_hand = game_env["nurse_hand"]
  tool_fornow = game_env["tool_fornow"]
  tool_locations = game_env["tool_locations"]
  current_step = game_env["current_step"]

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


# def tooldelivery_game_scene_names(game_env: Mapping[str, Any],
#                             cb_is_visible: Callable[[
#                                 co.DrawingObject], bool] = None
#                             ) -> List:
#   drawing_names = []
#   # drawing_names.append("Background")

#   for idx, _ in enumerate(game_env["Walls"]):
#     img_name = f"Wall{idx}"
#     if cb_is_visible is None or cb_is_visible(img_name):
#       drawing_names.append(img_name)

#   drawing_names.append(co.IMG_SCRUB)
#   drawing_names.append(co.IMG_SURGEON)
#   drawing_names.append(co.IMG_PATIENT)
#   drawing_names.append(co.IMG_PERF)
#   drawing_names.append(co.IMG_ANES)
#   drawing_names.append(co.IMG_CABINET)
#   drawing_names.append(co.IMG_STORAGE)

#   drawing_names.append(co.IMG_SCALPEL_STORED)
#   drawing_names.append(co.IMG_SCALPEL_PREPARED)
#   drawing_names.append(co.IMG_SUTURE_STORED)
#   drawing_names.append(co.IMG_SUTURE_PREPARED)
#   drawing_names.append(co.IMG_CIRCULATING)

#   return drawing_names
