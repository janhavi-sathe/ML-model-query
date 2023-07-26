from typing import Hashable, Mapping
import numpy as np
from ai_coach_domain.simulator import Simulator
from ai_coach_domain.tooldelivery.tooldelivery_v3_mdp import ToolDeliveryMDP_V3
from ai_coach_domain.tooldelivery.tooldelivery_v3_policy import (
    ToolDeliveryPolicy_V3)
import ai_coach_domain.tooldelivery.tooldelivery_v3_state_action as T3SA


class ToolDeliverySimulator(Simulator):
  CN = 0
  SN = 1
  AS = 2
  score = 0

  def __init__(self) -> None:
    super().__init__(0)

    self.mmdp = ToolDeliveryMDP_V3()
    self.policy = ToolDeliveryPolicy_V3(self.mmdp)
    self.num_brains = 2
    self.map_agent_2_brain = {}
    self.map_agent_2_brain[self.CN] = 0
    self.map_agent_2_brain[self.SN] = 1  # SN and AS maps to the same brain
    self.map_agent_2_brain[self.AS] = 1  # because they share their mental model

    self.p_CN_latent_scalpel = 0.5
    self.np_init_p_state = self.get_initial_distribution()

    self.grid_width = self.mmdp.num_x_grid + 5
    self.grid_height = 5
    self.max_steps = 200

    # background objects
    self.background = {
        "SN_pos_size": (6, 1, 1, 1),
        "AS_pos_size": (8, 1, 1, 1),
        "Table_pos_size": (5, 1, 1, 1),
        "Patient_pos_size": (6, 2, 3, 1),
        "Perfusionist_pos_size": (7, 3, 1, 1),
        "Anesthesiologist_pos_size": (9, 2, 1, 1),
        "Cabinet_pos_size": (*self.mmdp.cabinet_loc, 1, 1),
        "Storage_pos_size": (*self.mmdp.storage_loc, 1, 1),
        "Handover_pos_size": (*self.mmdp.handover_loc, 1, 1),
        "Walls": self.mmdp.walls
    }

    self.init_game()

  def init_game(self):
    self.reset_game()

  def get_initial_distribution(self):
    np_init_p_sScalpel_p = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SCALPEL_P].state_to_idx[
            T3SA.ToolLoc.SN]
    ]])
    np_init_p_sSuture_p = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SUTURE_P].state_to_idx[
            T3SA.ToolLoc.SN]
    ]])
    np_init_p_sScalpel_s = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SCALPEL_S].state_to_idx[
            T3SA.ToolLoc.STORAGE]
    ]])
    np_init_p_sSuture_s = np.array([[
        1., self.mmdp.dict_sTools_space[T3SA.ToolNames.SUTURE_S].state_to_idx[
            T3SA.ToolLoc.CABINET]
    ]])
    np_init_p_sPatient = np.array([[
        1., self.mmdp.sPatient_space.state_to_idx[T3SA.StatePatient.NO_INCISION]
    ]])
    np_init_p_sCNPos = np.array(
        [[1., self.mmdp.sCNPos_space.state_to_idx[self.mmdp.handover_loc]]])
    np_init_p_sAsked = np.array(
        [[1., self.mmdp.sAsked_space.state_to_idx[T3SA.StateAsked.NOT_ASKED]]])

    dict_init_p_state_idx = {}
    for p_sSa, sSa in np_init_p_sScalpel_p:
      for p_sFa, sFa in np_init_p_sSuture_p:
        for p_sSb, sSb in np_init_p_sScalpel_s:
          for p_sFb, sFb in np_init_p_sSuture_s:
            for p_sPat, sPat in np_init_p_sPatient:
              for p_sPos, sPos in np_init_p_sCNPos:
                for p_sAsk, sAsk in np_init_p_sAsked:
                  init_p = (p_sSa * p_sFa * p_sSb * p_sFb * p_sPat * p_sPos *
                            p_sAsk)
                  state_idx = self.mmdp.np_state_to_idx[sSa.astype(np.int32),
                                                        sFa.astype(np.int32),
                                                        sSb.astype(np.int32),
                                                        sFb.astype(np.int32),
                                                        sPat.astype(np.int32),
                                                        sPos.astype(np.int32),
                                                        sAsk.astype(np.int32)]
                  dict_init_p_state_idx[state_idx] = init_p

    np_init_p_state_idx = np.zeros((len(dict_init_p_state_idx), 2))
    iter_idx = 0
    for state_idx in dict_init_p_state_idx:
      np_next_p = dict_init_p_state_idx.get(state_idx)
      np_init_p_state_idx[iter_idx] = np_next_p, state_idx
      iter_idx += 1

    return np_init_p_state_idx

  def conv_action_idx_to_sim_action(self, action_idx):
    aCN, aSN, aAS = self.mmdp.np_idx_to_action[action_idx]
    action_cn = self.mmdp.aCN_space.idx_to_action[aCN]
    action_sn = self.mmdp.aSN_space.idx_to_action[aSN]
    action_as = self.mmdp.aAS_space.idx_to_action[aAS]
    return action_cn, action_sn, action_as

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:

    list_aidx = []
    tup_actions = tuple(
        [map_agent_2_action[idx] for idx in range(self.get_num_agents())])

    for idx, action in enumerate(tup_actions):
      list_aidx.append(
          self.mmdp.dict_factored_actionspace[idx].action_to_idx[action])
    action_idx = self.mmdp.np_action_to_idx[tuple(list_aidx)]
    self.history.append((self.state_idx, tup_actions, self.tup_latstate_idx))

    self.state_idx = self.mmdp.transition(self.state_idx, action_idx)
    self.tup_latstate_idx = self.get_latentstate(self.state_idx,
                                                 self.tup_latstate_idx)
    self.current_step += 1

  def event_input(self, agent: Hashable, event_type: Hashable, value=None):
    if agent == self.CN:
      if event_type in T3SA.ActionCN:
        self.CN_action = event_type
    elif agent == self.SN:
      if event_type in T3SA.ActionSN:
        self.SN_action = event_type
    elif agent == self.AS:
      if event_type in T3SA.ActionAS:
        self.AS_action = event_type

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:
    dict_agent_action = {}

    def get_action_impl(agent_id):
      np_p_action = self.policy.pi(
          self.state_idx,
          self.tup_latstate_idx[self.map_agent_2_brain[agent_id]])
      action_choice = np.random.choice(np_p_action[:, 1],
                                       1,
                                       p=np_p_action[:, 0])
      action_vector = self.mmdp.np_idx_to_action[action_choice[0].astype(
          np.int32)]
      return self.mmdp.dict_factored_actionspace[agent_id].idx_to_action[
          action_vector[agent_id]]

    if self.CN_action is not None:
      dict_agent_action[self.CN] = self.CN_action
    else:
      dict_agent_action[self.CN] = get_action_impl(self.CN)

    if self.SN_action is not None:
      dict_agent_action[self.SN] = self.SN_action
    else:
      dict_agent_action[self.SN] = get_action_impl(self.SN)

    if self.AS_action is not None:
      dict_agent_action[self.AS] = self.AS_action
    else:
      dict_agent_action[self.AS] = get_action_impl(self.AS)

    self.CN_action = None
    self.SN_action = None
    self.AS_action = None

    return dict_agent_action

  def reset_game(self):
    self.current_step = 0
    self.history = []
    self.state_idx = int(self.np_init_p_state[0][1])
    self.CN_action = None
    self.SN_action = None
    self.AS_action = None

    self.tup_latstate_idx = self.get_latentstate(self.state_idx, (None, None))

  def get_env_info(self):
    env_info = {}
    env_info.update(self.background)

    state_vec = self.mmdp.np_idx_to_state[self.state_idx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sPos, sAsk = state_vec

    s_scal_s = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_S].idx_to_state[sScal_s]
    s_sut_s = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_S].idx_to_state[sSut_s]
    s_sut_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]
    s_scal_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_P].idx_to_state[sScal_p]
    s_patient = self.mmdp.sPatient_space.idx_to_state[sPat]
    s_pos = self.mmdp.sCNPos_space.idx_to_state[sPos]
    s_ask = self.mmdp.sAsked_space.idx_to_state[sAsk]

    env_info["Scalpel_stored"] = s_scal_s
    env_info["Scalpel_prepared"] = s_scal_p
    env_info["Suture_stored"] = s_sut_s
    env_info["Suture_prepared"] = s_sut_p
    env_info["Patient_progress"] = s_patient
    env_info["CN_pos"] = s_pos
    env_info["Asked"] = s_ask
    env_info["current_step"] = self.current_step
    env_info["score"] = self.score

    return env_info

  def get_num_agents(self):
    return 3

  def get_changed_objects(self):
    pass

  def is_finished(self) -> bool:
    state_vector = self.mmdp.np_idx_to_state[self.state_idx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sPos, sAsk = state_vector

    s_scal_s = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SCALPEL_S].idx_to_state[sScal_s]
    s_sut_s = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_S].idx_to_state[sSut_s]
    s_sut_p = self.mmdp.dict_sTools_space[
        T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]
    s_patient = self.mmdp.sPatient_space.idx_to_state[sPat]

    if (s_scal_s == T3SA.ToolLoc.SN or s_sut_s == T3SA.ToolLoc.SN):
      return True

    if (s_sut_p == T3SA.ToolLoc.AS
        and s_patient == T3SA.StatePatient.NO_INCISION):
      return True

    return False

  def get_score(self):
    return -self.get_current_step()

  # #### mental state related methods
  def get_latstate_prior(self, obstate_idx):
    '''
        this method is valid only at the onset of StateAsked changing to 1.
        please do not use for other time steps
        '''

    np_cn_p_latent = None
    np_sn_p_latent = None

    state_vector = self.mmdp.np_idx_to_state[obstate_idx]
    sScal_p, sSut_p, sScal_s, sSut_s, sPat, sPos, sAsk = state_vector
    idx_to_state = self.mmdp.sAsked_space.idx_to_state
    if idx_to_state[sAsk] == T3SA.StateAsked.ASKED:
      s_scal_p = self.mmdp.dict_sTools_space[
          T3SA.ToolNames.SCALPEL_P].idx_to_state[sScal_p]
      s_sut_p = self.mmdp.dict_sTools_space[
          T3SA.ToolNames.SUTURE_P].idx_to_state[sSut_p]

      if s_scal_p == T3SA.ToolLoc.FLOOR:
        np_sn_p_latent = np.array([
            [1.0, T3SA.LatentState.SCALPEL.value],
        ])
      elif s_sut_p == T3SA.ToolLoc.FLOOR:
        np_sn_p_latent = np.array([[1.0, T3SA.LatentState.SUTURE.value]])
      elif s_scal_p == T3SA.ToolLoc.AS:
        np_sn_p_latent = np.array([[2.0 / 3.0, T3SA.LatentState.SCALPEL.value],
                                   [1.0 / 3.0, T3SA.LatentState.SUTURE.value]])
      elif s_sut_p == T3SA.ToolLoc.AS:
        np_sn_p_latent = np.array([[1.0 / 3.0, T3SA.LatentState.SCALPEL.value],
                                   [2.0 / 3.0, T3SA.LatentState.SUTURE.value]])

      if self.p_CN_latent_scalpel == 1.0:
        np_cn_p_latent = np.array([[1.0, T3SA.LatentState.SCALPEL.value]])
      elif self.p_CN_latent_scalpel == 0.0:
        np_cn_p_latent = np.array([[1.0, T3SA.LatentState.SUTURE.value]])
      else:
        np_cn_p_latent = np.array(
            [[self.p_CN_latent_scalpel, T3SA.LatentState.SCALPEL.value],
             [1.0 - self.p_CN_latent_scalpel, T3SA.LatentState.SUTURE.value]])

    return np_cn_p_latent, np_sn_p_latent

  def get_latentstate(self, obstate_idx, tup_cur_latent):
    if tup_cur_latent[0] is not None:
      return tup_cur_latent

    def choice_latent_state(np_prior_p_latent):
      lat_state = None
      if np_prior_p_latent[0][0] == 1.0:
        lat_state = np_prior_p_latent[0][1]
      else:
        lat_choice = np.random.choice(np_prior_p_latent[:, 1],
                                      1,
                                      p=np_prior_p_latent[:, 0])
        lat_state = lat_choice[0].astype(np.int32)
      return lat_state.astype(np.int32)

    np_cn_prior, np_sn_prior = self.get_latstate_prior(obstate_idx)
    if np_cn_prior is None or np_sn_prior is None:
      return None, None
    else:
      cn_lat = choice_latent_state(np_cn_prior)
      sn_lat = choice_latent_state(np_sn_prior)
      return cn_lat, sn_lat
