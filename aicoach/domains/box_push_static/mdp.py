from TMM.models.mdp import StateSpace
from TMM.domains.box_push.mdp import BoxPushAgentMDP, BoxPushTeamMDP
from TMM.domains.box_push import BoxState, EventType
from .transition import transition_alone_and_together


class BoxPushAgentMDP_AloneOrTogether(BoxPushAgentMDP):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_alone_and_together(box_states, a1_pos, a2_pos, a1_action,
                                         a2_action, self.boxes, self.goals,
                                         self.walls, self.drops, self.x_grid,
                                         self.y_grid)

  def get_possible_box_states(self):
    box_states = [(BoxState(idx), None) for idx in range(4)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


class BoxPushTeamMDP_AloneOrTogether(BoxPushTeamMDP):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_alone_and_together(box_states, a1_pos, a2_pos, a1_action,
                                         a2_action, self.boxes, self.goals,
                                         self.walls, self.drops, self.x_grid,
                                         self.y_grid)

  def get_possible_box_states(self):
    box_states = [(BoxState(idx), None) for idx in range(4)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    # a1 drops a box
    a1_drop = False
    if a1_pos in self.goals and act1 == EventType.UNHOLD:
      for bstate in box_states:
        if bstate[0] == BoxState.WithAgent1:
          a1_drop = True

    a2_drop = False
    if a2_pos in self.goals and act2 == EventType.UNHOLD:
      for bstate in box_states:
        if bstate[0] == BoxState.WithAgent2:
          a2_drop = True

    both_drop = False
    if (a1_pos in self.goals and a1_pos == a2_pos and act1 == EventType.UNHOLD
        and act2 == EventType.UNHOLD):
      for bstate in box_states:
        if bstate[0] == BoxState.WithBoth:
          both_drop = True

    reward = -1

    if a1_drop or a2_drop or both_drop:
      reward += 10

    return reward


class StaticBoxPushMDP(BoxPushTeamMDP_AloneOrTogether):

  def init_latentspace(self):
    latent_states = []
    latent_states.append(("alone", 0))
    latent_states.append(("together", 0))
    self.latent_space = StateSpace(latent_states)

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    with_a1 = -1
    with_a2 = -1
    with_both = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithAgent1:
        with_a1 = idx
      elif bstate[0] == BoxState.WithAgent2:
        with_a2 = idx
      elif bstate[0] == BoxState.WithBoth:
        with_both = idx

    move_actions = [
        EventType.UP, EventType.DOWN, EventType.LEFT, EventType.RIGHT
    ]

    panelty = -1
    if latent[0] == "together":
      if with_a1 >= 0 and act1 in move_actions:
        panelty += -5

      if with_a2 >= 0 and act2 in move_actions:
        panelty += -5

    if (with_a1 >= 0 and a1_pos in self.goals and act1 == EventType.UNHOLD
        and (a2_pos != a1_pos or act2 != EventType.HOLD)):
      return 100

    if (with_a2 >= 0 and a2_pos in self.goals and act2 == EventType.UNHOLD
        and (a1_pos != a2_pos or act1 != EventType.HOLD)):
      return 100

    if (with_both >= 0 and a1_pos in self.goals and act1 == EventType.UNHOLD
        and act2 == EventType.UNHOLD):
      return 100

    return panelty
