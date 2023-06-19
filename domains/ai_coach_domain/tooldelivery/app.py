import os
from typing import Hashable, Tuple
from stand_alone.app import AppInterface
from ai_coach_domain.tooldelivery.simulator import ToolDeliverySimulator
import ai_coach_domain.tooldelivery.tooldelivery_v3_state_action as T3SA


class ToolDeliveryApp(AppInterface):

  def __init__(self) -> None:
    super().__init__()
    self.image_dir = os.path.join(os.path.dirname(__file__), "images/")

  def _init_game(self):
    self.game = ToolDeliverySimulator()
    self.x_grid = self.game.grid_width
    self.y_grid = self.game.grid_height

  def _init_gui(self):
    self.main_window.title("Box Push")
    self.canvas_width = 600
    self.canvas_height = 300
    return super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    return super()._conv_key_to_agent_event(key_sym)

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    return super()._conv_mouse_to_agent_event(is_left, cursor_pos)

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()

  def _update_canvas_scene(self):
    data = self.game.get_env_info()

    walls = data["Walls"]
    SN_pos_size = data["SN_pos_size"]
    AS_pos_size = data["AS_pos_size"]
    Table_pos_size = data["Table_pos_size"]
    Patient_pos_size = data["Patient_pos_size"]
    Perfusionist_pos_size = data["Perfusionist_pos_size"]
    Anesthesiologist_pos_size = data["Anesthesiologist_pos_size"]
    Cabinet_pos_size = data["Cabinet_pos_size"]
    Storage_pos_size = data["Storage_pos_size"]
    Handover_pos_size = data["Handover_pos_size"]
    Scalpel_stored = data["Scalpel_stored"]
    Scalpel_prepared = data["Scalpel_prepared"]
    Suture_stored = data["Suture_stored"]
    Suture_prepared = data["Suture_prepared"]
    Patient_progress = data["Patient_progress"]
    CN_pos = data["CN_pos"]
    Asked = data["Asked"]

    x_unit = int(self.canvas_width / self.x_grid)
    y_unit = int(self.canvas_height / self.y_grid)

    self.clear_canvas()

    def draw_pos_size_obj(pos_size, obj_name, file_name):
      x, y, w, h = pos_size
      x_cen_cnvs = (x + 0.5 * w) * x_unit
      y_cen_cnvs = (y + 0.5 * h) * y_unit
      w_cnvs = w * x_unit
      h_cnvs = h * y_unit

      self.create_image(x_cen_cnvs, y_cen_cnvs, w_cnvs, h_cnvs, obj_name,
                        self.image_dir + file_name)

    def draw_pos_size_label(pos_size, text, color=None, font=None):
      x, y, w, h = pos_size
      x_txt_cnvs = (x + 0.5 * w) * x_unit
      y_txt_cnvs = (y + 0.8 * h) * y_unit
      self.create_text(x_txt_cnvs, y_txt_cnvs, text, color, font)

    # fixed objects
    for wall in walls:
      x, y = wall
      x = int(x) + 1
      self.create_line(x * x_unit, y * y_unit, x * x_unit, (y + 1) * y_unit,
                       "black")

    draw_pos_size_obj(SN_pos_size, "scrub", "scrub.png")
    draw_pos_size_obj(AS_pos_size, "surgeon", "surgeon.png")
    draw_pos_size_obj(Table_pos_size, "table", "table.png")
    draw_pos_size_obj(Patient_pos_size, "patient", "patient.png")
    draw_pos_size_obj(Perfusionist_pos_size, "perf", "human.png")
    draw_pos_size_obj(Anesthesiologist_pos_size, "anes", "human.png")
    draw_pos_size_obj(Cabinet_pos_size, "cabinet", "cabinet.png")
    draw_pos_size_obj(Storage_pos_size, "storage", "storage.png")

    draw_pos_size_label(SN_pos_size, "Scrub Nurse")
    draw_pos_size_label(AS_pos_size, "Surgeon")
    draw_pos_size_label(Patient_pos_size, "Patient")
    draw_pos_size_label(Perfusionist_pos_size, "Perf")
    draw_pos_size_label(Anesthesiologist_pos_size, "Anes")
    draw_pos_size_label(Cabinet_pos_size, "Cabinet", "white")
    draw_pos_size_label(Storage_pos_size, "Storage", "white")

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
        return (Storage_pos_size[0] + x_off, Storage_pos_size[1] + y_off,
                tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.CABINET:
        return (Cabinet_pos_size[0] + x_off, Cabinet_pos_size[1] + y_off,
                tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.FLOOR:
        return (6 + x_off, 3 + y_off, tool_w, tool_h)
      else:
        raise NotImplementedError

    draw_pos_size_obj((*CN_pos, 1, 1), "circulating", "circulating.png")
    draw_pos_size_obj(get_tool_pos_size(Scalpel_stored, 0.9, 0.1),
                      "scalpel_stored", "scalpel.png")
    draw_pos_size_obj(get_tool_pos_size(Scalpel_prepared, 1.0, 0.1),
                      "scalpel_prepared", "scalpel.png")
    draw_pos_size_obj(get_tool_pos_size(Suture_stored, 0.9, 0.1),
                      "suture_stored", "suture.png")
    draw_pos_size_obj(get_tool_pos_size(Suture_prepared, 1.0, 0.1),
                      "suture_prepared", "suture.png")

    draw_pos_size_label((*CN_pos, 1, 1), "CN")

    if len(self.game.history) != 0:
      _, tup_actions, tup_latents = self.game.history[-1]

      # action label
      self.create_text((SN_pos_size[0] + 0.5 * SN_pos_size[2]) * x_unit,
                       (SN_pos_size[1] + 1.0 * SN_pos_size[3]) * y_unit,
                       tup_actions[self.game.SN].name,
                       font=("Purisa", 10, 'italic'))
      self.create_text((AS_pos_size[0] + 0.5 * AS_pos_size[2]) * x_unit,
                       (AS_pos_size[1] + 1.0 * AS_pos_size[3]) * y_unit,
                       tup_actions[self.game.AS].name,
                       font=("Purisa", 10, 'italic'))

      # latent label
      if tup_latents[self.game.CN] is None:
        text_cn_latent = "None"
      else:
        text_cn_latent = T3SA.LatentState(tup_latents[self.game.CN]).name
      self.create_text((CN_pos[0] + 0.5) * x_unit, (CN_pos[1] - 0.2) * y_unit,
                       text_cn_latent)

      if tup_latents[self.game.SN] is None:
        text_sn_latent = "None"
      else:
        text_sn_latent = T3SA.LatentState(tup_latents[self.game.SN]).name
      self.create_text((SN_pos_size[0] + 0.5 * SN_pos_size[2]) * x_unit,
                       (SN_pos_size[1] - 0.2 * SN_pos_size[3]) * y_unit,
                       text_sn_latent)


if __name__ == "__main__":
  app = ToolDeliveryApp()
  app.run()
