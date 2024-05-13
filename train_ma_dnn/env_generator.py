from pettingzoo_domain.movers import Movers
from pettingzoo_domain.rescue import Rescue
from pettingzoo_domain.labor_division import (TwoTargetDyadLaborDivision,
                                              ThreeTargetDyadLaborDivision)
from pettingzoo.mpe import (simple_crypto_v3, simple_push_v3,
                            simple_adversary_v3, simple_speaker_listener_v4,
                            simple_spread_v3, simple_tag_v3)


def env_generator(config):
  '''
    return:
      fn_env_factory: a factory function that creates a pettingzoo env
      env_kwargs: a dictionary of kwargs for the env
  '''
  env_name = config.env_name
  if env_name == "ma_movers":
    return Movers, {}
  elif env_name == "ma_rescue":
    return Rescue, {}
  elif env_name == "LaborDivision2":
    return TwoTargetDyadLaborDivision, {}
  elif env_name == "LaborDivision3":
    return ThreeTargetDyadLaborDivision, {}
  # Multi Particle Environments (MPE)
  elif env_name == "simple_crypto":
    kwargs = {"continuous_actions": False}
    return simple_crypto_v3.parallel_env, kwargs
  elif env_name == "simple_push":
    kwargs = {"continuous_actions": False}
    return simple_push_v3.parallel_env, kwargs
  elif env_name == "simple_adversary":
    kwargs = {"continuous_actions": False}
    return simple_adversary_v3.parallel_env, kwargs
  elif env_name == "simple_speaker_listener":
    kwargs = {"continuous_actions": False}
    return simple_speaker_listener_v4.parallel_env, kwargs
  elif env_name == "simple_spread":
    kwargs = {"continuous_actions": False}
    return simple_spread_v3.parallel_env, kwargs
  elif env_name == "simple_tag":
    kwargs = {"continuous_actions": False}
    return simple_tag_v3.parallel_env, kwargs
