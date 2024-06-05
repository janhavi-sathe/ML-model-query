from pettingzoo_domain.labor_division import (TwoTargetDyadLaborDivision,
                                              ThreeTargetDyadLaborDivision,
                                              LDExpert_V2)
from pettingzoo_domain.po_movers_v2 import PO_Movers_V2, PO_Movers_AIAgent
from pettingzoo_domain.po_flood_v2 import PO_Flood_V2, PO_Flood_AIAgent
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_domain.box_push_v2.mdp import (MDP_Movers_Agent, MDP_Movers_Task)
from aic_domain.box_push_v2.policy import Policy_Movers
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue.policy import Policy_Rescue
from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task


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
  else:
    raise ValueError(f"{env_name} is not supported")

  return env, experts
