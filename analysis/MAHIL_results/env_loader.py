import os
from omegaconf import OmegaConf
from pettingzoo_domain.labor_division import (TwoTargetDyadLaborDivision,
                                              ThreeTargetDyadLaborDivision,
                                              LDExpert_V2)
from pettingzoo_domain.labor_division_v2 import (TwoTargetDyadLaborDivisionV2,
                                                 ThreeTargetDyadLaborDivisionV2,
                                                 LDv2Expert)
from pettingzoo_domain.po_movers_v2 import PO_Movers_V2, PO_Movers_AIAgent
from pettingzoo_domain.po_flood_v2 import PO_Flood_V2, PO_Flood_AIAgent
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.mdp import (MDP_Movers_Agent, MDP_Movers_Task)
from aic_domain.box_push_v2.policy import Policy_Movers
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue.policy import Policy_Rescue
from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task

from aic_ml.baselines.ma_ogail.model.agent import make_agent
from aic_ml.MAHIL.agent import make_mahil_agent
import bc_loader


def load_env(env_name):
  if env_name == "PO_Movers-v2":
    env = PO_Movers_V2()

    TEMPERATURE = 0.3
    game_map = MAP_MOVERS
    mdp_task = MDP_Movers_Task(**game_map)
    mdp_agent = MDP_Movers_Agent(**game_map)
    init_states = ([0] * len(game_map["boxes"]), game_map["a1_init"],
                   game_map["a2_init"])
    experts = {
        aname:
        PO_Movers_AIAgent(
            init_states, Policy_Movers(mdp_task, mdp_agent, TEMPERATURE, aname),
            env.possible_box_state, aname)
        for aname in env.possible_agents
    }

  elif env_name == "PO_Flood-v2":
    env = PO_Flood_V2()

    TEMPERATURE = 0.3
    game_map = MAP_RESCUE
    mdp_task = MDP_Rescue_Task(**game_map)
    mdp_agent = MDP_Rescue_Agent(**game_map)

    init_states = ([1] * len(game_map["work_locations"]), game_map["a1_init"],
                   game_map["a2_init"])

    experts = {
        aname:
        PO_Flood_AIAgent(init_states, aname,
                         Policy_Rescue(mdp_task, mdp_agent, TEMPERATURE, aname),
                         env.possible_locations)
        for aname in env.possible_agents
    }

  elif env_name == "LaborDivision2":
    env = TwoTargetDyadLaborDivision()
    experts = {
        aname: LDExpert_V2(env, env.tolerance, aname)
        for aname in env.possible_agents
    }
  elif env_name == "LaborDivision3":
    env = ThreeTargetDyadLaborDivision()
    experts = {
        aname: LDExpert_V2(env, env.tolerance, aname)
        for aname in env.possible_agents
    }
  elif env_name == "LaborDivision2-v2":
    env = TwoTargetDyadLaborDivisionV2()
    experts = {
        aname: LDv2Expert(env, env.tolerance, aname)
        for aname in env.possible_agents
    }
  elif env_name == "LaborDivision3-v2":
    env = ThreeTargetDyadLaborDivisionV2()
    experts = {
        aname: LDv2Expert(env, env.tolerance, aname)
        for aname in env.possible_agents
    }
  else:
    raise ValueError(f"{env_name} is not supported")

  return env, experts


def load_agent(env_name, alg_name, aname, sv, number):
  cur_dir = os.path.dirname(__file__)
  hri_results_dir = os.path.join(cur_dir, "result_hri", env_name)

  list_log_dir = []
  if env_name == "PO_Flood-v2":
    # iiql
    if alg_name == "iiql":
      list_log_dir.append("iiql/shortSeed1/2024-06-04_13-39-09")
      list_log_dir.append("iiql/shortSeed2/2024-06-04_14-11-58")
      list_log_dir.append("iiql/shortSeed3/2024-06-04_14-44-42")
    # mahil
    elif alg_name == "mahil":
      if sv == 0.0:
        list_log_dir.append("mahil/shortSeed1Sv0.0/2024-06-04_13-39-33")
        list_log_dir.append("mahil/shortSeed2Sv0.0/2024-06-04_14-21-07")
        list_log_dir.append("mahil/shortSeed3Sv0.0/2024-06-04_15-03-24")
      elif sv == 0.2:
        list_log_dir.append("mahil/shortSeed1Sv0.2/2024-06-04_13-39-36")
        list_log_dir.append("mahil/shortSeed2Sv0.2/2024-06-04_14-23-03")
        list_log_dir.append("mahil/shortSeed3Sv0.2/2024-06-04_15-06-04")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    # magail
    elif alg_name == "magail":
      list_log_dir.append("magail/tr2Seed1/2024-05-22_00-34-27")
      list_log_dir.append("magail/tr2Seed2/2024-05-22_02-36-12")
      list_log_dir.append("magail/tr2Seed3/2024-05-22_04-02-06")
    # maogail
    elif alg_name == "maogail":
      if sv == 0.0:
        list_log_dir.append("maogail/tr2Seed1Sv0.0/2024-05-22_00-37-31")
        list_log_dir.append("maogail/tr2Seed2Sv0.0/2024-05-22_05-33-20")
        list_log_dir.append("maogail/tr2Seed3Sv0.0/2024-05-22_09-09-57")
      elif sv == 0.2:
        list_log_dir.append("maogail/tr2Seed1Sv0.2/2024-05-22_00-38-40")
        list_log_dir.append("maogail/tr2Seed2Sv0.2/2024-05-22_05-34-02")
        list_log_dir.append("maogail/tr2Seed3Sv0.2/2024-05-22_09-10-09")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    elif alg_name == "bc":
      list_log_dir.append("bc/Seed1/2024-06-06_01-38-46")
      list_log_dir.append("bc/Seed2/2024-06-06_01-44-27")
      list_log_dir.append("bc/Seed3/2024-06-06_01-50-05")
    else:
      raise ValueError(f"Algorithm {alg_name} is not supported")
  elif env_name == "PO_Movers-v2":
    # iiql
    if alg_name == "iiql":
      list_log_dir.append("iiql/shortSeed1/2024-06-04_13-39-16")
      list_log_dir.append("iiql/shortSeed2/2024-06-04_14-21-18")
      list_log_dir.append("iiql/shortSeed3/2024-06-04_15-03-33")
    # mahil
    elif alg_name == "mahil":
      if sv == 0.0:
        list_log_dir.append("mahil/shortSeed1Sv0.0/2024-06-04_13-39-39")
        list_log_dir.append("mahil/shortSeed2Sv0.0/2024-06-04_14-33-38")
        list_log_dir.append("mahil/shortSeed3Sv0.0/2024-06-04_15-28-11")
      elif sv == 0.2:
        list_log_dir.append("mahil/shortSeed1Sv0.2/2024-06-04_13-39-43")
        list_log_dir.append("mahil/shortSeed2Sv0.2/2024-06-04_14-31-14")
        list_log_dir.append("mahil/shortSeed3Sv0.2/2024-06-04_15-24-07")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    # magail
    elif alg_name == "magail":
      list_log_dir.append("magail/tr2Seed1/2024-05-22_00-33-00")
      list_log_dir.append("magail/tr2Seed2/2024-05-22_02-34-21")
      list_log_dir.append("magail/tr2Seed3/2024-05-22_03-59-38")
    # maogail
    elif alg_name == "maogail":
      if sv == 0.0:
        list_log_dir.append("maogail/Seed1Sv0.0/2024-05-21_22-57-30")
        list_log_dir.append("maogail/Seed2Sv0.0/2024-05-22_04-17-31")
        list_log_dir.append("maogail/Seed3Sv0.0/2024-05-22_08-07-42")
      elif sv == 0.2:
        list_log_dir.append("maogail/Seed1Sv0.2/2024-05-21_22-57-44")
        list_log_dir.append("maogail/Seed2Sv0.2/2024-05-22_04-17-05")
        list_log_dir.append("maogail/Seed3Sv0.2/2024-05-22_08-06-09")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    elif alg_name == "bc":
      list_log_dir.append("bc/Seed1/2024-06-06_01-38-52")
      list_log_dir.append("bc/Seed2/2024-06-06_01-45-24")
      list_log_dir.append("bc/Seed3/2024-06-06_01-51-53")
    else:
      raise ValueError(f"Algorithm {alg_name} is not supported")
  elif env_name == "LaborDivision2":
    # iiql
    if alg_name == "iiql":
      list_log_dir.append("mahil/Seed1/2024-05-21_22-45-27")
      list_log_dir.append("mahil/Seed2/2024-05-22_00-51-49")
      list_log_dir.append("mahil/Seed3/2024-05-22_02-52-45")
    # mahil
    elif alg_name == "mahil":
      if sv == 0.0:
        list_log_dir.append("mahil/Seed1Sv0.0/2024-05-21_22-47-02")
        list_log_dir.append("mahil/Seed2Sv0.0/2024-05-22_00-58-51")
        list_log_dir.append("mahil/Seed3Sv0.0/2024-05-22_02-58-41")
      elif sv == 0.2:
        list_log_dir.append("mahil/Seed1Sv0.2/2024-05-21_22-47-20")
        list_log_dir.append("mahil/Seed2Sv0.2/2024-05-22_00-58-25")
        list_log_dir.append("mahil/Seed3Sv0.2/2024-05-22_02-58-13")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    # magail
    elif alg_name == "magail":
      list_log_dir.append("magail/Seed1/2024-05-21_22-40-55")
      list_log_dir.append("magail/Seed2/2024-05-21_23-27-38")
      list_log_dir.append("magail/Seed3/2024-05-22_00-32-51")
    # maogail
    elif alg_name == "maogail":
      if sv == 0.0:
        list_log_dir.append("maogail/Seed1Sv0.0/2024-05-21_22-54-54")
        list_log_dir.append("maogail/Seed2Sv0.0/2024-05-22_02-54-39")
        list_log_dir.append("maogail/Seed3Sv0.0/2024-05-22_06-10-47")
      elif sv == 0.2:
        list_log_dir.append("maogail/Seed1Sv0.2/2024-05-21_22-55-07")
        list_log_dir.append("maogail/Seed2Sv0.2/2024-05-22_02-54-43")
        list_log_dir.append("maogail/Seed3Sv0.2/2024-05-22_06-09-57")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    else:
      raise ValueError(f"Algorithm {alg_name} is not supported")
  elif env_name == "LaborDivision3":
    # iiql
    if alg_name == "iiql":
      list_log_dir.append("mahil/Seed1/2024-05-21_22-45-41")
      list_log_dir.append("mahil/Seed2/2024-05-22_00-52-34")
      list_log_dir.append("mahil/Seed3/2024-05-22_02-52-59")
    # mahil
    elif alg_name == "mahil":
      if sv == 0.0:
        list_log_dir.append("mahil/Seed1Sv0.0/2024-05-21_22-47-52")
        list_log_dir.append("mahil/Seed2Sv0.0/2024-05-22_01-03-22")
        list_log_dir.append("mahil/Seed3Sv0.0/2024-05-22_03-03-47")
      elif sv == 0.2:
        list_log_dir.append("mahil/Seed1Sv0.2/2024-05-21_22-48-56")
        list_log_dir.append("mahil/Seed2Sv0.2/2024-05-22_01-04-17")
        list_log_dir.append("mahil/Seed3Sv0.2/2024-05-22_03-03-58")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    # magail
    elif alg_name == "magail":
      list_log_dir.append("magail/Seed1/2024-05-21_22-42-18")
      list_log_dir.append("magail/Seed2/2024-05-21_23-33-47")
      list_log_dir.append("magail/Seed3/2024-05-22_00-40-55")
    # maogail
    elif alg_name == "maogail":
      if sv == 0.0:
        list_log_dir.append("maogail/Seed1Sv0.0/2024-05-21_22-56-12")
        list_log_dir.append("maogail/Seed2Sv0.0/2024-05-22_03-05-07")
        list_log_dir.append("maogail/Seed3Sv0.0/2024-05-22_06-22-21")
      elif sv == 0.2:
        list_log_dir.append("maogail/Seed1Sv0.2/2024-05-21_22-56-26")
        list_log_dir.append("maogail/Seed2Sv0.2/2024-05-22_03-05-17")
        list_log_dir.append("maogail/Seed3Sv0.2/2024-05-22_06-22-31")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    else:
      raise ValueError(f"Algorithm {alg_name} is not supported")
  elif env_name == "LaborDivision2-v2":
    # iiql
    if alg_name == "iiql":
      list_log_dir.append("iiql/Seed1/2024-06-05_08-56-35")
      list_log_dir.append("iiql/Seed2/2024-06-05_11-23-35")
      list_log_dir.append("iiql/Seed3/2024-06-05_13-30-15")
    # mahil
    elif alg_name == "mahil":
      if sv == 0.0:
        list_log_dir.append("mahil/Seed1Sv0.0/2024-06-05_09-00-51")
        list_log_dir.append("mahil/Seed2Sv0.0/2024-06-05_11-40-02")
        list_log_dir.append("mahil/Seed3Sv0.0/2024-06-05_13-52-37")
      elif sv == 0.2:
        list_log_dir.append("mahil/Seed1Sv0.2/2024-06-05_09-01-10")
        list_log_dir.append("mahil/Seed2Sv0.2/2024-06-05_11-39-17")
        list_log_dir.append("mahil/Seed3Sv0.2/2024-06-05_13-52-12")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    # magail
    elif alg_name == "magail":
      list_log_dir.append("magail/Seed1/2024-06-05_08-59-58")
      list_log_dir.append("magail/Seed2/2024-06-05_09-40-12")
      list_log_dir.append("magail/Seed3/2024-06-05_10-21-07")
    # maogail
    elif alg_name == "maogail":
      if sv == 0.0:
        list_log_dir.append("maogail/Seed1Sv0.0/2024-06-05_09-02-09")
        list_log_dir.append("maogail/Seed2Sv0.0/2024-06-05_10-28-22")
        list_log_dir.append("maogail/Seed3Sv0.0/2024-06-05_11-53-45")
      elif sv == 0.2:
        list_log_dir.append("maogail/Seed1Sv0.2/2024-06-05_09-02-26")
        list_log_dir.append("maogail/Seed2Sv0.2/2024-06-05_10-28-44")
        list_log_dir.append("maogail/Seed3Sv0.2/2024-06-05_11-54-17")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    elif alg_name == "bc":
      list_log_dir.append("bc/reSeed1/2024-06-06_14-46-58")
      list_log_dir.append("bc/reSeed2/2024-06-06_14-49-47")
      list_log_dir.append("bc/reSeed3/2024-06-06_14-52-38")
      # list_log_dir.append("bc/Seed1/2024-06-06_01-38-03")
      # list_log_dir.append("bc/Seed2/2024-06-06_01-44-02")
      # list_log_dir.append("bc/Seed3/2024-06-06_01-50-05")
    else:
      raise ValueError(f"Algorithm {alg_name} is not supported")
  elif env_name == "LaborDivision3-v2":
    # iiql
    if alg_name == "iiql":
      list_log_dir.append("iiql/Seed1/2024-06-05_08-59-33")
      list_log_dir.append("iiql/Seed2/2024-06-05_11-32-16")
      list_log_dir.append("iiql/Seed3/2024-06-05_13-38-38")
    # mahil
    elif alg_name == "mahil":
      if sv == 0.0:
        list_log_dir.append("mahil/Seed1Sv0.0/2024-06-05_09-01-27")
        list_log_dir.append("mahil/Seed2Sv0.0/2024-06-05_11-46-11")
        list_log_dir.append("mahil/Seed3Sv0.0/2024-06-05_14-05-35")
      elif sv == 0.2:
        list_log_dir.append("mahil/Seed1Sv0.2/2024-06-05_09-01-41")
        list_log_dir.append("mahil/Seed2Sv0.2/2024-06-05_11-45-14")
        list_log_dir.append("mahil/Seed3Sv0.2/2024-06-05_14-04-14")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    # magail
    elif alg_name == "magail":
      list_log_dir.append("magail/Seed1/2024-06-05_09-00-27")
      list_log_dir.append("magail/Seed2/2024-06-05_09-41-10")
      list_log_dir.append("magail/Seed3/2024-06-05_10-22-05")
    # maogail
    elif alg_name == "maogail":
      if sv == 0.0:
        list_log_dir.append("maogail/Seed1Sv0.0/2024-06-05_09-02-46")
        list_log_dir.append("maogail/Seed2Sv0.0/2024-06-05_10-40-41")
        list_log_dir.append("maogail/Seed3Sv0.0/2024-06-05_12-18-46")
      elif sv == 0.2:
        list_log_dir.append("maogail/Seed1Sv0.2/2024-06-05_09-03-09")
        list_log_dir.append("maogail/Seed2Sv0.2/2024-06-05_10-41-10")
        list_log_dir.append("maogail/Seed3Sv0.2/2024-06-05_12-20-25")
      else:
        raise ValueError(f"Supervision {sv} is not supported")
    elif alg_name == "bc":
      list_log_dir.append("bc/reSeed1/2024-06-06_14-47-52")
      list_log_dir.append("bc/reSeed2/2024-06-06_14-50-42")
      list_log_dir.append("bc/reSeed3/2024-06-06_14-53-35")
      # list_log_dir.append("bc/Seed1/2024-06-06_01-38-41")
      # list_log_dir.append("bc/Seed2/2024-06-06_01-44-48")
      # list_log_dir.append("bc/Seed3/2024-06-06_01-50-53")
    else:
      raise ValueError(f"Algorithm {alg_name} is not supported")
  else:
    raise ValueError(f"{env_name} is not supported")

  log_dir = os.path.join(hri_results_dir, list_log_dir[number - 1])

  config_path = os.path.join(log_dir, "log/config.yaml")
  config = OmegaConf.load(config_path)

  n_traj = int(config.n_traj)
  n_label = 0 if alg_name in ["iiql", "magail"] else int(n_traj * sv)
  model_name = env_name + f"_n{n_traj}_l{n_label}_best_{aname}"
  model_path = os.path.join(log_dir, f"model/{model_name}")

  env, _ = load_env(env_name)
  env.reset()

  if alg_name == "iiql" or alg_name == "mahil":
    agent = make_mahil_agent(config, env, aname)
  elif alg_name == "magail":
    agent = make_agent(config, env, aname, False)
  elif alg_name == "maogail":
    agent = make_agent(config, env, aname, True)
  elif alg_name == "bc":
    agent = bc_loader.load_bc_agent(config, env, aname, model_path)
  else:
    raise ValueError(f"Algorithm {alg_name} is not supported")
  agent.load(model_path)

  return agent
