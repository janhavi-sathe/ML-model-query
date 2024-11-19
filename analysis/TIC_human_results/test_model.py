import numpy as np
import pickle
import random
from aic_domain.agent import BTILCachedPolicy
from aic_domain.box_push_v2.agent import BoxPushAIAgent_BTIL
from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.mdp import (MDP_Movers_Agent, MDP_Movers_Task)
from aic_domain.rescue.simulator import RescueSimulator
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
from aic_domain.rescue.agent import AIAgent_Rescue_BTIL
from aic_domain.rescue import (Location, E_Type, E_EventType)


def load_models(model_dir, domain):
  if domain == "movers":
    v_value_file = "movers_160_0,30_150_merged_v_values_learned.pickle"
    policy1_file = "movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy"
    policy2_file = "movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy"
    tx1_file = "movers_btil_dec_tx_human_FTTT_160_0,30_a1.npy"
    tx2_file = "movers_btil_dec_tx_human_FTTT_160_0,30_a2.npy"

    mask = (False, True, True, True)
    game_map = MAP_MOVERS
    mdp_task = MDP_Movers_Task(**game_map)
    mdp_agent = MDP_Movers_Agent(**game_map)
    game = BoxPushSimulatorV2(0)
    BASE_CLASS_AGENT = BoxPushAIAgent_BTIL
  elif domain == "flood":
    v_value_file = "rescue_2_160_0,30_30_merged_v_values_learned.pickle"
    policy1_file = "rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy"
    policy2_file = "rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy"
    tx1_file = "rescue_2_btil_dec_tx_human_FTTT_160_0,30_a1.npy"
    tx2_file = "rescue_2_btil_dec_tx_human_FTTT_160_0,30_a2.npy"

    mask = (False, True, True, True)
    game_map = MAP_RESCUE
    mdp_task = MDP_Rescue_Task(**game_map)
    mdp_agent = MDP_Rescue_Agent(**game_map)
    game = RescueSimulator()
    BASE_CLASS_AGENT = AIAgent_Rescue_BTIL
  else:
    raise ValueError(f"Intervention page for {domain} is not implemented.")

  with open(model_dir + v_value_file, 'rb') as handle:
    v_values = pickle.load(handle)

  np_policy1 = np.load(model_dir + policy1_file)
  np_policy2 = np.load(model_dir + policy2_file)
  list_np_pi = [np_policy1, np_policy2]

  np_tx1 = np.load(model_dir + tx1_file)
  np_tx2 = np.load(model_dir + tx2_file)
  list_np_tx = [np_tx1, np_tx2]

  # return list_np_pi, list_np_tx, v_values
  agents = []
  for idx, np_policy in enumerate(list_np_pi):
    policy = BTILCachedPolicy(np_policy, mdp_task, idx, mdp_agent.latent_space)
    agent = BASE_CLASS_AGENT(list_np_tx[idx], mask, policy, idx)
    agents.append(agent)

  game.init_game(**game_map)
  game.set_autonomous_agent(*agents)
  return game, agents


def run_movers_with_intent(model_dir,
                           intent,
                           n_repeat,
                           max_step=150,
                           box_state=None):
  game, agents = load_models(model_dir, "movers")
  game.max_steps = max_step

  dict_per_boxstate = {}
  list_success_steps = []
  list_wrong_subgoals = []
  for _ in range(n_repeat):
    game.reset_game()

    if box_state is not None:
      cur_box_state = tuple(box_state)
    else:
      cur_box_state = np.random.choice([0, 4], 3)
      if intent[0] == "pickup":
        cur_box_state[intent[1]] = 0
      else:
        cur_box_state[np.random.randint(3)] = 3
      cur_box_state = tuple(cur_box_state)

    # for stats
    n_b, n_b_s = dict_per_boxstate.get(cur_box_state, (0, 0))
    dict_per_boxstate[cur_box_state] = (n_b + 1, n_b_s)

    possible_subgoals = game.boxes + game.goals

    # set state & latent
    game.box_states = cur_box_state
    if intent[0] == "pickup":
      pos_subgoal = game.boxes[intent[1]]
      game.agent_2.init_latent(game.get_current_state())
    elif intent[0] == "goal":
      bidx = game.box_states.index(3)
      game.a1_pos = game.a2_pos = game.boxes[bidx]
      pos_subgoal = game.goals[intent[1]]

    while not game.is_finished():
      game.agent_1.set_latent(intent)  # fix latent
      if intent[0] == "goal":
        game.agent_2.set_latent(intent)
      map_agent_2_action = game.get_joint_action()
      game.take_a_step(map_agent_2_action)
      if game.a1_pos in possible_subgoals:
        if game.a1_pos == pos_subgoal:
          list_success_steps.append(game.current_step)
          n_b, n_b_s = dict_per_boxstate[cur_box_state]
          dict_per_boxstate[cur_box_state] = (n_b, n_b_s + 1)
          break
        elif intent[0] == "pickup":
          list_wrong_subgoals.append(game.a1_pos)
          break
        elif intent[0] == "goal" and game.a1_pos != game.boxes[bidx]:
          list_wrong_subgoals.append(game.a1_pos)
          break

  return list_success_steps, dict_per_boxstate, list_wrong_subgoals


def run_flood_with_intent(model_dir,
                          intent,
                          n_repeat,
                          max_step=30,
                          work_state=None):
  game, agents = load_models(model_dir, "flood")
  game.max_steps = max_step

  # intent
  # 0: 'City Hall', 1: 'Bridge 1', 2: 'Campsite', 3: 'Bridge 2'

  dict_per_workstate = {}
  list_success_steps = []
  list_wrong_subgoals = []
  for _ in range(n_repeat):
    game.reset_game()

    if work_state is not None:
      cur_work_state = tuple(work_state)
    else:
      bg_state = random.choice([(0, 1), (1, 0), (1, 1)])
      ch_state = np.random.choice([0, 1])
      cs_state = np.random.choice([0, 1])
      cur_work_state = [ch_state, bg_state[0], cs_state, bg_state[1]]
      if intent in [1, 3]:
        cur_work_state[1] = cur_work_state[3] = 1
      else:
        cur_work_state[intent] = 1
      cur_work_state = tuple(cur_work_state)

    game.work_states = cur_work_state

    # for stats
    n_w, n_w_s = dict_per_workstate.get(cur_work_state, (0, 0))
    dict_per_workstate[cur_work_state] = (n_w + 1, n_w_s)

    possible_subgoals = game.work_locations

    # init agents & states
    pos_subgoal = game.work_locations[intent]
    game.agent_2.init_latent(game.get_current_state())
    # possible_init_loc = [Location(E_Type.Place, 2)]
    # for widx, ws in enumerate(cur_work_state):
    #   if ws == 0:
    #     possible_init_loc.append(game.work_locations[widx])

    # game.a1_pos = random.choice(possible_init_loc)  # random start

    while not game.is_finished():
      game.agent_1.set_latent(intent)  # fix latent
      map_agent_2_action = game.get_joint_action()
      # don't let agent2 rescue
      if map_agent_2_action[game.AGENT2] == E_EventType.Rescue:
        map_agent_2_action[game.AGENT2] = E_EventType.Stay

      game.take_a_step(map_agent_2_action)
      if game.a1_pos in possible_subgoals:
        if game.a1_pos == pos_subgoal:
          list_success_steps.append(game.current_step)
          n_w, n_w_s = dict_per_workstate[cur_work_state]
          dict_per_workstate[cur_work_state] = (n_w, n_w_s + 1)
          break
        else:
          list_wrong_subgoals.append(game.a1_pos)
          break

  return list_success_steps, dict_per_workstate, list_wrong_subgoals


def get_intent_behavior_result(model_dir, domain):
  for xidx in range(4):
    if domain == "movers":
      if xidx != 3:
        intent = ("pickup", xidx)
      else:
        intent = ("goal", 0)

      print("--------", intent)

      n_repeat = 1000
      list_success_steps, dict_per_state, wrong_arrival = (
          run_movers_with_intent(model_dir, intent, n_repeat, max_step=20))
    elif domain == "flood":
      print("--------", xidx)
      n_repeat = 1000
      list_success_steps, dict_per_state, wrong_arrival = (
          run_flood_with_intent(model_dir,
                                xidx,
                                n_repeat,
                                max_step=20,
                                work_state=(1, 1, 1, 1)))

    n_success = len(list_success_steps)
    rate = n_success / n_repeat
    if n_success == 0:
      print("No success")
    else:
      med = np.median(list_success_steps)
      mean = np.mean(list_success_steps)
      q1 = np.percentile(list_success_steps, 25)
      q3 = np.percentile(list_success_steps, 75)
      n_wrong = len(wrong_arrival)
      wrong_rate = n_wrong / n_repeat
      n_nowhere = n_repeat - n_success - n_wrong
      nowhere_rate = n_nowhere / n_repeat
      print(f"Success rate: {n_success} / {n_repeat} (={rate})")
      print(f"Wrong place rate: {n_wrong} / {n_repeat} (={wrong_rate})")
      print(f"Nowhere rate: {n_nowhere} / {n_repeat} (={nowhere_rate})")
      print(f"Percential: {q1}, {med}, {q3} (mean: {mean})")
      print(dict_per_state)


def get_task_result(model_dir, domain, n_repeat):
  game, agents = load_models(model_dir, domain)
  if domain == "movers":
    game.max_steps = 150
  elif domain == "flood":
    game.max_steps = 30
  else:
    raise ValueError(f"{domain} is not implemented.")

  list_scores = []
  for _ in range(n_repeat):
    game.reset_game()

    while not game.is_finished():
      map_agent_2_action = game.get_joint_action()
      game.take_a_step(map_agent_2_action)

    list_scores.append(game.get_score())

  med = np.median(list_scores)
  mean = np.mean(list_scores)
  q1 = np.percentile(list_scores, 25)
  q3 = np.percentile(list_scores, 75)
  print(f"Percential: {q1}, {med}, {q3} (mean: {mean})")


if __name__ == "__main__":

  model_dir = ("/home/sangwon/Projects/ai_coach/" +
               "web_app_v2/web_experiment/exp_intervention/model_data/")

  # get_intent_behavior_result(model_dir, "movers")
  get_intent_behavior_result(model_dir, "flood")
  # get_task_result(model_dir, "movers", 1000)
  # get_task_result(model_dir, "flood", 1000)
