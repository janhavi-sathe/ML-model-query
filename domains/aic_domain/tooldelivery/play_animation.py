import os
import tkinter
import tkinter.scrolledtext
import tkinter.filedialog
from PIL import Image, ImageTk
from aic_domain.tooldelivery.utils import read_sample
from aic_domain.tooldelivery.tooldelivery_v3_env import ToolDeliveryEnv_V3
import aic_domain.tooldelivery.tooldelivery_v3_state_action as T3SA


class ToolDeliveryPlayer():

  def __init__(self):
    self.gui = tkinter.Tk()
    self.gui.title("Surgical tool delivery")
    self.env = ToolDeliveryEnv_V3(use_policy=False)

    self.num_grid_x = 10
    self.num_grid_y = 5
    self.step_x = 50
    self.step_y = 50

    # background objects
    self.SN = (6, 1, 1, 1)  # x, y, w, h
    self.AS = (8, 1, 1, 1)  # x, y, w, h
    self.Tbl = (5, 1, 1, 1)
    self.Pat = (6, 2, 3, 1)
    self.Prf = (7, 3, 1, 1)
    self.Ans = (9, 2, 1, 1)

    self.Cab = (2, 4, 1, 1)
    self.Sto = (0, 4, 1, 1)
    self.Han = (4, 1, 1, 1)

    self.step_length = 0.5  # in sec
    self.trajectory = None
    self.latent_states = None
    self.images = {}
    self.image_dir = os.path.join(os.path.dirname(__file__), "images/")

    # initialize
    self.init_gui()

    self.CN_coord = None
    self.CN_item = None
    self.Scalpel_p = None
    self.Suture_p = None
    self.Scalpel_s = None
    self.Suture_s = None
    self.time_step = 0
    self.auto_play = False
    self.sn_action = None
    self.as_action = None
    self.cn_mental = None
    self.sn_mental = None
    self.cn_text = None

  def init_gui(self):
    # buttons
    self.btn_auto = tkinter.Button(self.gui,
                                   text="Autoplay",
                                   command=self.callback_autoplay_btn)
    self.btn_auto.grid(row=0, column=0)
    self.btn_step = tkinter.Button(self.gui,
                                   text="Step",
                                   command=self.callback_step_btn)
    self.btn_step.grid(row=0, column=1)
    self.btn_open = tkinter.Button(self.gui,
                                   text="Open",
                                   command=self.callback_open_btn)
    self.btn_open.grid(row=0, column=2)

    # labels
    widget = tkinter.Label(self.gui,
                           text="Latent: ",
                           justify=tkinter.LEFT,
                           width=20)
    widget.grid(row=0, column=3)
    self.label_latent = tkinter.Label(self.gui,
                                      text="None, None",
                                      justify=tkinter.LEFT,
                                      width=20)
    self.label_latent.grid(row=0, column=4)

    # canvas
    canvas_width = self.num_grid_x * self.step_x
    canvas_height = self.num_grid_y * self.step_y
    self.canvas = tkinter.Canvas(self.gui,
                                 width=canvas_width,
                                 height=canvas_height,
                                 highlightbackground="black")
    self.canvas.grid(row=1, columnspan=5)
    self.draw_grid_line()
    self.draw_fixed_objects()

    # text box
    self.wgtextbox = tkinter.scrolledtext.ScrolledText(self.gui,
                                                       width=70,
                                                       height=10)

    self.wgtextbox.grid(row=2, columnspan=5)
    self.wgtextbox.config(state='disabled')

  def draw_grid_line(self):
    NUM_CN_GRID_X = 5
    NUM_CN_GRID_Y = 5
    canvas_width = NUM_CN_GRID_X * self.step_x
    canvas_height = NUM_CN_GRID_Y * self.step_y
    # for index in range(1, NUM_CN_GRID_Y):
    #     self.canvas.create_line(
    #         0, self.step_y * index, canvas_width, self.step_y * index,
    #         dash=(4, 2))
    # for index in range(1, NUM_CN_GRID_X + 1):
    #     self.canvas.create_line(
    #         self.step_x * index, 0, self.step_x * index, canvas_height,
    #         dash=(4, 2))
    # wall
    self.canvas.create_line(self.step_x * 2,
                            0,
                            self.step_x * 2,
                            1 * self.step_y,
                            width=3)
    self.canvas.create_line(self.step_x * 2,
                            2 * self.step_y,
                            self.step_x * 2,
                            canvas_height,
                            width=3)

  def create_image(self,
                   x_grid_cen,
                   y_grid_cen,
                   width,
                   height,
                   obj_name,
                   file_name=None):
    x_pos_st = x_grid_cen * self.step_x
    y_pos_st = y_grid_cen * self.step_y

    if obj_name not in self.images:
      img = Image.open(file_name)
      f_w, f_h = img.size
      w_ratio = self.step_x * width / f_w
      h_ratio = self.step_y * height / f_h
      new_size_x = 0
      new_size_y = 0
      if w_ratio < h_ratio:
        new_size_x = int(self.step_x * width)
        new_size_y = int(f_h * w_ratio)
      else:
        new_size_x = int(f_w * h_ratio)
        new_size_y = int(self.step_y * height)

      img = img.resize((new_size_x, new_size_y), Image.ANTIALIAS)
      img = ImageTk.PhotoImage(img)
      self.images[obj_name] = img
    else:
      img = self.images[obj_name]

    return self.canvas.create_image(x_pos_st,
                                    y_pos_st,
                                    anchor=tkinter.CENTER,
                                    image=img)

  def draw_fixed_objects(self):
    # self.create_oval(
    #     self.SN[0], self.SN[1], self.SN[2], self.SN[3], "red")

    self.create_image(self.SN[0] + 0.5, self.SN[1] + 0.5, self.SN[2],
                      self.SN[3], "scrub", self.image_dir + "scrub.png")
    self.create_image(self.AS[0] + 0.5, self.AS[1] + 0.5, self.AS[2],
                      self.AS[3], "surgeon", self.image_dir + "surgeon.png")
    self.create_image(self.Prf[0] + 0.5, self.Prf[1] + 0.5, self.Prf[2],
                      self.Prf[3], "perf", self.image_dir + "human.png")
    self.create_image(self.Ans[0] + 0.5, self.Ans[1] + 0.5, self.Ans[2],
                      self.Ans[3], "anes", self.image_dir + "human.png")
    self.create_image(self.Pat[0] + 0.5 * self.Pat[2],
                      self.Pat[1] + 0.5 * self.Pat[3], self.Pat[2] * 1.2,
                      self.Pat[3] * 1.2, "patient",
                      self.image_dir + "patient.png")
    self.create_image(self.Tbl[0] + 0.5, self.Tbl[1] + 0.5, self.Tbl[2],
                      self.Tbl[3], "table", self.image_dir + "table.png")
    self.create_image(self.Cab[0] + 0.5, self.Cab[1] + 0.5, self.Cab[2],
                      self.Cab[3], "cabinet", self.image_dir + "cabinet.png")
    self.create_image(self.Sto[0] + 0.5, self.Sto[1] + 0.5, self.Sto[2],
                      self.Sto[3], "storage", self.image_dir + "storage.png")

    # self.create_rectangle(
    #     self.Han[0], self.Han[1], self.Han[2], self.Han[3], "light gray")
    self.create_text(self.Pat[0] + 0.5 * self.Pat[2], self.Pat[1] + 0.8,
                     "Patient")
    self.create_text(self.SN[0] + 0.5, self.SN[1] + 0.8, "SN", "black")
    self.create_text(self.AS[0] + 0.5, self.AS[1] + 0.8, "Surgeon", "black")
    self.create_text(self.Prf[0] + 0.5, self.Prf[1] + 0.8, "PERF", "black")
    self.create_text(self.Ans[0] + 0.5, self.Ans[1] + 0.8, "ANES", "black")
    self.create_text(self.Cab[0] + 0.5,
                     self.Cab[1] + 0.8,
                     "Cabinet",
                     "white",
                     font=("Purisa", 10))
    self.create_text(self.Sto[0] + 0.5,
                     self.Sto[1] + 0.8,
                     "Storage",
                     "white",
                     font=("Purisa", 10))

  def draw_specific_time_step(self, time_step):
    state_idx, action_idx = self.trajectory[time_step]
    state_vector = self.env.mmdp.np_idx_to_state[state_idx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sCNPos, sAsk = state_vector
    self.CN_coord = self.env.mmdp.sCNPos_space.idx_to_state[sCNPos]
    s_asked = self.env.mmdp.sAsked_space.idx_to_state[sAsk]
    aCN, aSN, aAS = self.env.mmdp.np_idx_to_action[action_idx]
    action_sn = self.env.mmdp.aSN_space.idx_to_action[aSN]
    action_as = self.env.mmdp.aAS_space.idx_to_action[aAS]

    def get_tool_coord(tool_nm, sTool):
      s_tool = (self.env.mmdp.dict_sTools_space[tool_nm].idx_to_state[sTool])
      tool_w = 0.5
      tool_h = 0.5
      off = 0.1
      if s_tool == T3SA.ToolLoc.AS:
        return (self.AS[0] + off, self.AS[1] + off, tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.SN:
        return (self.Tbl[0], self.Tbl[1] + off, tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.CN:
        return (self.CN_coord[0] + off, self.CN_coord[1] + off, tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.STORAGE:
        return (self.Sto[0] + off, self.Sto[1] + off, tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.CABINET:
        return (self.Cab[0] + off, self.Cab[1] + off, tool_w, tool_h)
      elif s_tool == T3SA.ToolLoc.FLOOR:
        return (6 + off, 3 + off, tool_w, tool_h)

      assert False

    if self.CN_item is not None:
      self.canvas.delete(self.CN_item)
    if self.Scalpel_p is not None:
      self.canvas.delete(self.Scalpel_p)
    if self.Suture_p is not None:
      self.canvas.delete(self.Suture_p)
    if self.Scalpel_s is not None:
      self.canvas.delete(self.Scalpel_s)
    if self.Suture_s is not None:
      self.canvas.delete(self.Suture_s)

    if self.sn_action is not None:
      self.canvas.delete(self.sn_action)
    if self.as_action is not None:
      self.canvas.delete(self.as_action)
    if self.cn_mental is not None:
      self.canvas.delete(self.cn_mental)
    if self.sn_mental is not None:
      self.canvas.delete(self.sn_mental)
    if self.cn_text is not None:
      self.canvas.delete(self.cn_text)

    self.CN_item = self.create_image(self.CN_coord[0] + 0.5,
                                     self.CN_coord[1] + 0.5, 1, 1,
                                     "circulating",
                                     self.image_dir + "circulating.png")

    coord = get_tool_coord(T3SA.ToolNames.SCALPEL_P, sScal_p)
    self.Scalpel_p = self.create_image(coord[0] + 0.8, coord[1] + 0.0, coord[2],
                                       coord[3], "scalpel_p",
                                       self.image_dir + "scalpel.png")
    coord = get_tool_coord(T3SA.ToolNames.SUTURE_P, sSut_p)
    self.Suture_p = self.create_image(coord[0] + 0.9, coord[1] + 0.0, coord[2],
                                      coord[3], "suture_p",
                                      self.image_dir + "suture.png")
    coord = get_tool_coord(T3SA.ToolNames.SCALPEL_S, sScal_s)
    self.Scalpel_s = self.create_image(coord[0] + 0.8, coord[1] + 0.0, coord[2],
                                       coord[3], "scalpel_s",
                                       self.image_dir + "scalpel.png")
    coord = get_tool_coord(T3SA.ToolNames.SUTURE_S, sSut_s)
    self.Suture_s = self.create_image(coord[0] + 0.9, coord[1] + 0.0, coord[2],
                                      coord[3], "suture_s",
                                      self.image_dir + "suture.png")

    self.cn_text = self.create_text(self.CN_coord[0] + 0.5,
                                    self.CN_coord[1] + 0.8, "CN", "black")
    self.sn_action = self.create_text(self.SN[0] + 0.5,
                                      self.SN[1] + 1.0,
                                      action_sn.name,
                                      font=("Purisa", 10, 'italic'))
    self.as_action = self.create_text(self.AS[0] + 0.5,
                                      self.AS[1] + 1.0,
                                      action_as.name,
                                      font=("Purisa", 10, 'italic'))

    if self.env.is_initiated_state(state_idx):
      cn_text = ('None' if self.latent_states[0] is None else T3SA.LatentState(
          self.latent_states[0]).name)
      sn_text = ('None' if self.latent_states[1] is None else T3SA.LatentState(
          self.latent_states[1]).name)

      self.cn_mental = self.create_text(self.CN_coord[0] + 0.5,
                                        self.CN_coord[1] - 0.2,
                                        "\'" + cn_text + "\'")
      self.sn_mental = self.create_text(self.SN[0] + 0.5, self.SN[1] - 0.2,
                                        "\'" + sn_text + "\'")

    if self.env.is_initiated_state(state_idx):
      self.change_label(str(self.latent_states))
    else:
      self.change_label(str((None, None)))

  def periodic_call(self):
    if self.auto_play:
      self.take_one_step()
      self.gui_after = self.gui.after(int(self.step_length * 1000),
                                      self.periodic_call)

  def control_btn_state(self, state):
    self.btn_open.config(state=state)
    self.btn_step.config(state=state)

  def callback_autoplay_btn(self):
    if self.trajectory is None:
      return

    self.auto_play = not self.auto_play
    if self.auto_play:
      self.btn_auto.config(text="Stop")
      self.control_btn_state(state=tkinter.DISABLED)
      self.periodic_call()
    else:
      self.btn_auto.config(text="Autoplay")
      self.control_btn_state(state=tkinter.NORMAL)
      self.gui.after_cancel(self.gui_after)

  def take_one_step(self):
    if self.time_step >= len(self.trajectory):
      self.time_step = 0

    if self.time_step == 0:
      self.wgtextbox.configure(state=tkinter.NORMAL)
      self.wgtextbox.delete('1.0', tkinter.END)
      self.wgtextbox.configure(state=tkinter.DISABLED)
      self.wgtextbox.yview(tkinter.END)
      text_tmp = "Scalpel_p; Suture_p; Scalpel_s; Suture_s;"
      text_tmp += "Patient; CN; Request" + "\n"
      text_tmp += "CN_action; SN_action; AS_action" + "\n"
      text_tmp += "---------------------------------------------"
      self.append_conversation(text_tmp)

    self.draw_specific_time_step(self.time_step)
    sidx, aidx = self.trajectory[self.time_step]

    state_text = self.state_string(sidx)
    action_text = self.action_string(aidx)
    self.append_conversation(state_text + '\n' + action_text)

    self.time_step = self.time_step + 1

  def action_string(self, aidx):
    if aidx < 0:
      return "None"
    action_vector = self.env.mmdp.np_idx_to_action[aidx]
    aCN, aSN, aAS = action_vector
    cn_action = self.env.mmdp.aCN_space.idx_to_action[aCN]
    sn_action = self.env.mmdp.aSN_space.idx_to_action[aSN]
    as_action = self.env.mmdp.aAS_space.idx_to_action[aAS]

    action_text = cn_action.name + "; "
    action_text += sn_action.name + "; "
    action_text += as_action.name

    return action_text

  def state_string(self, sidx):
    state_vector = self.env.mmdp.np_idx_to_state[sidx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sCNPos, sAsk = state_vector
    scal_p = self.env.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_P].idx_to_state[sScal_p]
    sut_p = self.env.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]
    scal_s = self.env.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_S].idx_to_state[sScal_s]
    sut_s = self.env.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_S].idx_to_state[sSut_s]
    pat = self.env.mmdp.sPatient_space.idx_to_state[sPat]
    pos = self.env.mmdp.sCNPos_space.idx_to_state[sCNPos]
    ask = self.env.mmdp.sAsked_space.idx_to_state[sAsk]

    state_text = scal_p.name + "; "
    state_text += sut_p.name + "; "
    state_text += scal_s.name + "; "
    state_text += sut_s.name + "; "
    state_text += pat.name + "; "
    state_text += str(pos) + "; "
    state_text += ask.name

    return state_text

  def callback_step_btn(self):
    if self.trajectory is None:
      return
    if self.auto_play:
      return
    self.take_one_step()

  def callback_open_btn(self):
    if self.auto_play:
      return
    file_name = tkinter.filedialog.askopenfilename()
    if not file_name:
      return
    traj, tuple_lat = read_sample(file_name)
    self.trajectory = traj
    self.latent_states = tuple_lat
    self.time_step = 0
    self.take_one_step()
    pass

  def change_label(self, txt):
    self.label_latent.config(text=txt)

  def append_conversation(self, msg):
    self.wgtextbox.configure(state=tkinter.NORMAL)
    self.wgtextbox.insert(tkinter.END, msg + '\n')
    self.wgtextbox.configure(state=tkinter.DISABLED)
    self.wgtextbox.yview(tkinter.END)

  def update_scene(self):
    self.gui.update()

  def create_rectangle(self, x_grid, y_grid, w_grid, h_grid, color):
    x_pos_st = x_grid * self.step_x
    x_pos_ed = (x_grid + w_grid) * self.step_x
    y_pos_st = y_grid * self.step_y
    y_pos_ed = (y_grid + h_grid) * self.step_y

    return self.canvas.create_rectangle(x_pos_st,
                                        y_pos_st,
                                        x_pos_ed,
                                        y_pos_ed,
                                        fill=color)

  def create_oval(self, x_grid, y_grid, w_grid, h_grid, color):
    x_pos_st = x_grid * self.step_x
    x_pos_ed = (x_grid + w_grid) * self.step_x
    y_pos_st = y_grid * self.step_y
    y_pos_ed = (y_grid + h_grid) * self.step_y

    return self.canvas.create_oval(x_pos_st,
                                   y_pos_st,
                                   x_pos_ed,
                                   y_pos_ed,
                                   fill=color)

  def create_text(self,
                  x_grid,
                  y_grid,
                  txt,
                  color="black",
                  font=("Purisa", 10, 'bold')):
    return self.canvas.create_text(x_grid * self.step_x,
                                   y_grid * self.step_y,
                                   text=txt,
                                   fill=color,
                                   font=font)

  def create_triangle(self, x_grid, y_grid, w_grid, h_grid, color):
    x_pos_1 = x_grid * self.step_x
    y_pos_1 = y_grid * self.step_y

    x_pos_2 = (x_grid + w_grid) * self.step_x
    y_pos_2 = y_grid * self.step_y

    x_pos_3 = (x_pos_1 + x_pos_2) / 2
    y_pos_3 = (y_grid + h_grid) * self.step_y
    points = [x_pos_1, y_pos_1, x_pos_3, y_pos_3, x_pos_2, y_pos_2]

    return self.canvas.create_polygon(points, fill=color)

  def create_inverted_triangle(self, x_grid, y_grid, w_grid, h_grid, color):
    x_pos_1 = x_grid * self.step_x
    y_pos_1 = (y_grid + h_grid) * self.step_y

    x_pos_2 = (x_grid + w_grid) * self.step_x
    y_pos_2 = (y_grid + h_grid) * self.step_y

    x_pos_3 = (x_pos_1 + x_pos_2) / 2
    y_pos_3 = y_grid * self.step_y
    points = [x_pos_1, y_pos_1, x_pos_2, y_pos_2, x_pos_3, y_pos_3]

    return self.canvas.create_polygon(points, fill=color)

  def run(self):
    self.gui.mainloop()


if __name__ == "__main__":
  # gui = tkinter.Tk()
  player = ToolDeliveryPlayer()
  player.run()
