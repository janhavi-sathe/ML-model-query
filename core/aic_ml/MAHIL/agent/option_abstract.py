import abc
import torch
import numpy as np
from aic_ml.baselines.IQLearn.utils.utils import one_hot
from omegaconf import DictConfig


class AbstractPolicyLeaner(abc.ABC):

  def __init__(self, config: DictConfig):
    self.gamma = config.gamma
    self.device = torch.device(config.device)
    self.actor = None
    self.clip_grad_val = config.clip_grad_val
    self.num_critic_update = config.num_critic_update
    self.num_actor_update = config.num_actor_update

  def conv_input(self, batch_input, is_onehot_needed, dimension):
    if is_onehot_needed:
      if not isinstance(batch_input, torch.Tensor):
        batch_input = torch.tensor(
            batch_input, dtype=torch.float).reshape(-1).to(self.device)
      else:
        batch_input = batch_input.reshape(-1)
      # TODO: used a trick to handle initial/unobserved values.
      #       may need to find a better way later
      batch_input = one_hot(batch_input, dimension + 1)
      batch_input = batch_input[:, :-1]
    else:
      if not isinstance(batch_input, torch.Tensor):
        batch_input = torch.tensor(np.array(batch_input).reshape(-1, dimension),
                                   dtype=torch.float).to(self.device)

    return batch_input

  def conv_tuple_input(self, tup_batch, tup_is_onehot_needed, tup_dimension):
    list_batch = []
    for idx in range(len(tup_batch)):
      batch = self.conv_input(tup_batch[idx], tup_is_onehot_needed[idx],
                              tup_dimension[idx])
      list_batch.append(batch)

    # concat
    batch_input = torch.cat(list_batch, dim=1)

    return batch_input

  @abc.abstractmethod
  def reset_optimizers(self, config: DictConfig):
    pass

  @abc.abstractmethod
  def train(self, training=True):
    pass

  @property
  @abc.abstractmethod
  def alpha(self):
    pass

  @property
  @abc.abstractmethod
  def critic_net(self):
    pass

  @property
  @abc.abstractmethod
  def critic_target_net(self):
    pass

  @abc.abstractmethod
  def choose_action(self, tup_obs, option, sample=False):
    pass

  @abc.abstractmethod
  def critic(self, tup_obs, option, action, both=False):
    pass

  @abc.abstractmethod
  def getV(self, tup_obs, option):
    pass

  @abc.abstractmethod
  def get_targetV(self, tup_obs, option):
    pass

  @abc.abstractmethod
  def update(self, tup_obs, option, action, tup_next_obs, next_option, reward,
             done, logger, step):
    pass

  @abc.abstractmethod
  def update_critic(self, tup_obs, option, action, tup_next_obs, next_option,
                    reward, done, logger, step):
    pass

  @abc.abstractmethod
  def save(self, path, suffix=""):
    pass

  @abc.abstractmethod
  def load(self, path):
    pass

  @abc.abstractmethod
  def log_probs(self, tup_obs, action):
    pass
