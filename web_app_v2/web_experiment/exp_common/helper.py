import os
import time

import web_experiment.exp_common.canvas_objects as co


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
