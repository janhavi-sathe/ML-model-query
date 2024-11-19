from typing import List
import os
import stable_baselines3 as sb3
import gym
import random
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.util import logger as sb3_logger
from pettingzoo.utils.env import ParallelEnv  # noqa: F401
import wandb
import omegaconf
from aic_ml.MAHIL.helper.utils import load_trajectories


class BCActor:

  def __init__(self, actor_critic: ActorCriticPolicy) -> None:
    self.actor_critic = actor_critic
    self.PREV_LATENT = None
    self.PREV_AUX = None

  def choose_action(self, obs, prew_lat, sample=False, **kwargs):
    action, _ = self.actor_critic.predict(obs, not sample)

    return None, action

  def load(self, model_path):
    # do nothing
    pass


def load_bc_agent(config, env, aname, model_path):

  dummy_scheduler = lambda _: torch.finfo(torch.float32).max
  actor_critic = ActorCriticPolicy(
      observation_space=env.observation_space(aname),
      action_space=env.action_space(aname),
      lr_schedule=dummy_scheduler,
      net_arch=dict(pi=config.hidden_policy, vf=config.hidden_critic),
      activation_fn=nn.ReLU)
  actor_critic.load(model_path)
  return BCActor(actor_critic)
