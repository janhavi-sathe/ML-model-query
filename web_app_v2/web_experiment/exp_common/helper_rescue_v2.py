from dataclasses import dataclass
from typing import Mapping, Any, List, Tuple, Callable, Sequence
import os
import time
import numpy as np
from ai_coach_domain.rescue_v2 import (Place, Route, Location, E_Type,
                                       PlaceName, Work, is_work_done,
                                       T_Connections)
import web_experiment.exp_common.canvas_objects as co
from web_experiment.exp_common.helper import DrawInfo

RESCUE_V2_PLACE_DRAW_INFO = {
    PlaceName.Fire_stateion:
    DrawInfo(co.IMG_FIRE_STATION, (0.07, -0.06), (0.10, 0.10),
             0,
             circles=[(0.07, -0.03, 0.1), (0, 0, 0.06)]),
    PlaceName.Police_station:
    DrawInfo(co.IMG_POLICE_STATION, (-0.1, -0.03), (0.10, 0.10),
             0,
             circles=[(-0.07, -0.02, 0.1), (0, 0, 0.06)]),
    PlaceName.Campsite:
    DrawInfo(co.IMG_CAMPSITE, (-0.05, -0.05), (0.13, 0.09),
             0,
             circles=[(0.01, -0.17, 0.2), (0, 0, 0.06)]),
    PlaceName.City_hall:
    DrawInfo(co.IMG_CITY_HALL, (0, -0.08), (0.10, 0.08),
             0,
             circles=[(0, -0.03, 0.11)]),
    PlaceName.Mall:
    DrawInfo(co.IMG_MALL, (0.06, 0.05), (0.10, 0.10),
             0,
             circles=[(0.06, 0.03, 0.12), (0, 0, 0.06)]),
    PlaceName.Hospital:
    DrawInfo(co.IMG_HOSPITAL, (0.05, -0.06), (0.10, 0.09),
             0,
             circles=[(0.03, -0.06, 0.1), (0, 0, 0.06)]),
}


def location_2_coord_v2(loc: Location, places: Sequence[Place],
                        routes: Sequence[Route]):
  if loc.type == E_Type.Place:
    return places[loc.id].coord
  else:
    route_id = loc.id  # type: int
    route = routes[route_id]  # type: Route
    idx = loc.index
    return route.coords[idx]


def rescue_v2_game_scene(
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
  connections = game_env["connections"]  # type: Mapping[int, T_Connections]
  game_objs = []

  def add_obj(obj):
    if cb_is_visible is None or cb_is_visible(obj):
      game_objs.append(obj)

  obj = co.Circle("hi", (0, 0), 2, fill_color="green", alpha=0.8)
  add_obj(obj)

  return game_objs


def rescue_v2_game_scene_names(
    game_env: Mapping[str, Any],
    cb_is_visible: Callable[[str], bool] = None) -> List:

  drawing_names = []

  def add_obj_name(obj_name):
    if cb_is_visible is None or cb_is_visible(obj_name):
      drawing_names.append(obj_name)

  add_obj_name("hi")

  return drawing_names
