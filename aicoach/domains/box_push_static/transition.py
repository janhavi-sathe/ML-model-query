from TMM.domains.box_push import (BoxState, EventType, conv_box_state_2_idx)
from TMM.domains.box_push.transition import (get_box_idx_impl,
                                             get_moved_coord_impl,
                                             hold_state_impl,
                                             update_dropped_box_state_impl,
                                             is_opposite_direction)


def transition_alone_and_together(box_states: list, a1_pos, a2_pos, a1_act,
                                  a2_act, box_locations: list, goals: list,
                                  walls: list, drops: list, x_bound, y_bound):
  # num_goals = len(goals)
  num_drops = len(drops)

  # methods
  def get_box_idx(coord):
    return get_box_idx_impl(coord, box_states, a1_pos, a2_pos, box_locations,
                            goals, drops)

  def get_moved_coord(coord, action, box_states=None):
    return get_moved_coord_impl(coord, action, x_bound, y_bound, walls,
                                box_states, a1_pos, a2_pos, box_locations,
                                goals, drops)

  def hold_state():
    return hold_state_impl(box_states, drops, goals)

  def update_dropped_box_state(boxidx, coord, box_states_new):
    return update_dropped_box_state_impl(boxidx, coord, box_states_new,
                                         box_locations, drops, goals)

  P_MOVE = 0.7
  list_next_env = []
  hold = hold_state()
  # both do not hold anything
  if hold == "None":
    box_states_new = list(box_states)
    if (a1_act == EventType.HOLD and a2_act == EventType.HOLD
        and a1_pos == a2_pos):
      bidx = get_box_idx(a1_pos)
      if bidx >= 0:
        state = (BoxState.WithBoth, None)
        box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)

      list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
    else:
      a1_pos_new = a1_pos
      if a1_act == EventType.HOLD:
        bidx = get_box_idx(a1_pos)
        if bidx >= 0:
          state = (BoxState.WithAgent1, None)
          box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act)

      a2_pos_new = a2_pos
      if a2_act == EventType.HOLD:
        bidx = get_box_idx(a2_pos)
        if bidx >= 0:
          state = (BoxState.WithAgent2, None)
          box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act)

      list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))
  # both hold the same box
  elif hold == "Both":
    bidx = get_box_idx(a1_pos)
    assert bidx >= 0
    # invalid case
    if a1_pos != a2_pos:
      list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      return list_next_env

    box_states_new = list(box_states)
    # both try to drop the box
    if a1_act == EventType.UNHOLD and a2_act == EventType.UNHOLD:
      # if succeeded to drop
      if update_dropped_box_state(bidx, a1_pos, box_states_new):
        list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
      # if failed to drop due to invalid location
      else:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    num_drops)
        box_states_new2 = list(box_states)
        box_states_new2[bidx] = conv_box_state_2_idx(
            (BoxState.WithAgent2, None), num_drops)

        list_next_env.append((0.5, box_states_new, a1_pos, a2_pos))
        list_next_env.append((0.5, box_states_new2, a1_pos, a2_pos))
    # only agent1 try to unhold
    elif a1_act == EventType.UNHOLD:
      box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                  num_drops)
      a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
      if a2_pos_new == a2_pos:
        list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
      else:
        list_next_env.append((P_MOVE, box_states_new, a1_pos, a2_pos_new))
        list_next_env.append((1.0 - P_MOVE, box_states_new, a1_pos, a2_pos))
    elif a2_act == EventType.UNHOLD:
      box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                  num_drops)
      a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
      if a1_pos_new == a1_pos:
        list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
      else:
        list_next_env.append((P_MOVE, box_states_new, a1_pos_new, a2_pos))
        list_next_env.append((1.0 - P_MOVE, box_states_new, a1_pos, a2_pos))
    else:
      if (is_opposite_direction(a1_act, a2_act)
          or (a1_act == EventType.STAY and a2_act == EventType.STAY)):
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      elif a1_act == a2_act:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        list_next_env.append((1.0, box_states, a1_pos_new, a1_pos_new))
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
        if a1_pos_new == a2_pos_new:
          list_next_env.append((1.0, box_states, a1_pos_new, a1_pos_new))
        else:
          list_next_env.append((0.5, box_states, a1_pos_new, a1_pos_new))
          list_next_env.append((0.5, box_states, a2_pos_new, a2_pos_new))
  elif hold == "Each":
    box_states_new = list(box_states)
    a1_pos_new = a1_pos
    a2_pos_new = a2_pos
    # if more than one remains on the same grid
    if (a1_act in [EventType.UNHOLD, EventType.STAY]
        or a2_act in [EventType.UNHOLD, EventType.STAY]):
      p_a1_success = 0
      if a1_act == EventType.UNHOLD:
        bidx = get_box_idx(a1_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a1_pos, box_states_new)
        p_a1_success = 1
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

      p_a2_success = 0
      if a2_act == EventType.UNHOLD:
        bidx = get_box_idx(a2_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a2_pos, box_states_new)
        p_a2_success = 1
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

      p_ss = p_a1_success * p_a2_success
      p_sf = p_a1_success * (1 - p_a2_success)
      p_fs = (1 - p_a1_success) * p_a2_success
      p_ff = (1 - p_a1_success) * (1 - p_a2_success)
      if p_ss > 0:
        list_next_env.append((p_ss, box_states_new, a1_pos_new, a2_pos_new))
      if p_sf > 0:
        list_next_env.append((p_sf, box_states_new, a1_pos_new, a2_pos))
      if p_fs > 0:
        list_next_env.append((p_fs, box_states_new, a1_pos, a2_pos_new))
      if p_ff > 0:
        list_next_env.append((p_ff, box_states_new, a1_pos, a2_pos))
    # when both try to move
    else:
      agent_dist = (abs(a1_pos[0] - a2_pos[0]) + abs(a1_pos[1] - a2_pos[1]))
      if agent_dist > 2:
        p_a1_success = 0
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

        p_a2_success = 0
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

        p_ss = p_a1_success * p_a2_success
        p_sf = p_a1_success * (1 - p_a2_success)
        p_fs = (1 - p_a1_success) * p_a2_success
        p_ff = (1 - p_a1_success) * (1 - p_a2_success)
        if p_ss > 0:
          list_next_env.append((p_ss, box_states, a1_pos_new, a2_pos_new))
        if p_sf > 0:
          list_next_env.append((p_sf, box_states, a1_pos_new, a2_pos))
        if p_fs > 0:
          list_next_env.append((p_fs, box_states, a1_pos, a2_pos_new))
        if p_ff > 0:
          list_next_env.append((p_ff, box_states, a1_pos, a2_pos))
      elif agent_dist == 2:
        p_a1_success = 0
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

        p_a2_success = 0
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

        p_ss = p_a1_success * p_a2_success
        p_sf = p_a1_success * (1 - p_a2_success)
        p_fs = (1 - p_a1_success) * p_a2_success
        p_ff = (1 - p_a1_success) * (1 - p_a2_success)

        if a1_pos_new == a2_pos_new:
          if p_ss > 0:
            list_next_env.append((p_ss * 0.5, box_states, a1_pos_new, a2_pos))
            list_next_env.append((p_ss * 0.5, box_states, a1_pos, a2_pos_new))
        else:
          if p_ss > 0:
            list_next_env.append((p_ss, box_states, a1_pos_new, a2_pos_new))

        if p_sf > 0:
          list_next_env.append((p_sf, box_states, a1_pos_new, a2_pos))
        if p_fs > 0:
          list_next_env.append((p_fs, box_states, a1_pos, a2_pos_new))
        if p_ff > 0:
          list_next_env.append((p_ff, box_states, a1_pos, a2_pos))
      else:  # agent_dst == 1
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
        if a1_pos_new != a1_pos and a2_pos_new != a2_pos:
          list_next_env.append(
              (P_MOVE * P_MOVE, box_states, a1_pos_new, a2_pos_new))
          list_next_env.append(
              (P_MOVE * (1 - P_MOVE), box_states, a1_pos_new, a2_pos))
          list_next_env.append(
              (P_MOVE * (1 - P_MOVE), box_states, a1_pos, a2_pos_new))
          list_next_env.append(
              ((1 - P_MOVE) * (1 - P_MOVE), box_states, a1_pos, a2_pos))
        elif a1_pos_new != a1_pos:
          a2_pos_new2 = get_moved_coord(a2_pos, a2_act)
          if a2_pos_new2 == a1_pos:
            list_next_env.append(
                (P_MOVE * P_MOVE, box_states, a1_pos_new, a1_pos))
            list_next_env.append(
                (P_MOVE * (1 - P_MOVE), box_states, a1_pos_new, a2_pos))
            list_next_env.append(((1 - P_MOVE), box_states, a1_pos, a2_pos))
          else:
            list_next_env.append((P_MOVE, box_states, a1_pos_new, a2_pos))
            list_next_env.append((1 - P_MOVE, box_states, a1_pos, a2_pos))
        elif a2_pos_new != a2_pos:
          a1_pos_new2 = get_moved_coord(a1_pos, a1_act)
          if a1_pos_new2 == a2_pos:
            list_next_env.append(
                (P_MOVE * P_MOVE, box_states, a2_pos, a2_pos_new))
            list_next_env.append(
                ((1 - P_MOVE) * P_MOVE, box_states, a1_pos, a2_pos_new))
            list_next_env.append(((1 - P_MOVE), box_states, a1_pos, a2_pos))
          else:
            list_next_env.append((P_MOVE, box_states, a1_pos, a2_pos_new))
            list_next_env.append((1 - P_MOVE, box_states, a1_pos, a2_pos))
        else:
          list_next_env.append((1.0, box_states, a1_pos, a2_pos))
  # only a1 holds a box
  elif hold == "A1":
    if a2_act == EventType.HOLD and a1_pos == a2_pos:
      box_states_new = list(box_states)
      bidx = get_box_idx(a1_pos)
      assert bidx >= 0
      if a1_act == EventType.UNHOLD:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                    num_drops)
      else:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithBoth, None),
                                                    num_drops)
      list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
    else:
      box_states_new = list(box_states)
      p_a1_success = 0
      a1_pos_new = a1_pos
      if a1_act == EventType.UNHOLD:
        bidx = get_box_idx(a1_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a1_pos, box_states_new)
        p_a1_success = 1.0
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

      a2_pos_new = a2_pos
      if a2_act == EventType.HOLD:
        bidx = get_box_idx(a2_pos)
        if bidx >= 0:
          box_states_new[bidx] = conv_box_state_2_idx(
              (BoxState.WithAgent2, None), num_drops)
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act)

      if p_a1_success > 0:
        list_next_env.append(
            (p_a1_success, box_states_new, a1_pos_new, a2_pos_new))
      if 1 - p_a1_success > 0:
        list_next_env.append(
            (1 - p_a1_success, box_states_new, a1_pos, a2_pos_new))
  # only a2 holds a box
  else:  # hold == "A2":
    if a1_act == EventType.HOLD and a1_pos == a2_pos:
      box_states_new = list(box_states)
      bidx = get_box_idx(a2_pos)
      assert bidx >= 0
      if a2_act == EventType.UNHOLD:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    num_drops)
      else:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithBoth, None),
                                                    num_drops)
      list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
    else:
      box_states_new = list(box_states)
      p_a2_success = 0
      a2_pos_new = a2_pos
      if a2_act == EventType.UNHOLD:
        bidx = get_box_idx(a2_pos)
        assert bidx >= 0
        update_dropped_box_state(bidx, a2_pos, box_states_new)
        p_a2_success = 1.0
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

      a1_pos_new = a1_pos
      if a1_act == EventType.HOLD:
        bidx = get_box_idx(a1_pos)
        if bidx >= 0:
          box_states_new[bidx] = conv_box_state_2_idx(
              (BoxState.WithAgent1, None), num_drops)
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act)

      if p_a2_success > 0:
        list_next_env.append(
            (p_a2_success, box_states_new, a1_pos_new, a2_pos_new))
      if 1 - p_a2_success > 0:
        list_next_env.append(
            (1 - p_a2_success, box_states_new, a1_pos_new, a2_pos))

  return list_next_env
