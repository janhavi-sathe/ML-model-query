from aic_domain.tooldelivery.tooldelivery_v3_policy import ToolDeliveryPolicy_V3
from aic_domain.tooldelivery.simulator import ToolDeliverySimulator

from aic_domain.rescue_v2.maps import MAP_RESCUE
from web_experiment.models import db, User
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.helper import (get_file_name,
                                              store_user_label_locally)
from web_experiment.exp_common.page_tooldelivery_base import (ToolDeliveryGamePageBase)

TEMPERATURE = 0.3


class ToolDeliveryGamePage(ToolDeliveryGamePageBase):

  def __init__(self,
               manual_latent_selection,
               auto_prompt: bool = True,
               prompt_on_change: bool = True,
               prompt_freq: int = 5) -> None:
    super().__init__(manual_latent_selection, MAP_RESCUE, auto_prompt,
                     prompt_on_change, prompt_freq)

    self._TEAMMATE_POLICY = ToolDeliveryPolicy_V3

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''
    super()._on_game_finished(user_game_data)

    game = user_game_data.get_game_ref()  # type: ToolDeliverySimulator
    user = user_game_data.data[Exp1UserData.USER]
    user_id = user.userid

    # save trajectory
    save_path = user_game_data.data[Exp1UserData.SAVE_PATH]
    session_name = user_game_data.data[Exp1UserData.SESSION_NAME]
    file_name = get_file_name(save_path, user_id, session_name)
    header = game.__class__.__name__ + "-" + session_name + "\n"
    header += "User ID: %s\n" % (str(user_id), )
    header += str(self._GAME_MAP)
    game.save_history(file_name, header)

    user_label_path = user_game_data.data[Exp1UserData.USER_LABEL_PATH]
    user_labels = user_game_data.data[Exp1UserData.USER_LABELS]
    store_user_label_locally(user_label_path, user_id, session_name,
                             user_labels)

    # update score
    best_score = user.best_c

    if best_score < game.score:
      user = User.query.filter_by(userid=user_id).first()
      user.best_c = game.score
      db.session.commit()
      user_game_data.data[Exp1UserData.USER] = user

    # move to next page
    user_game_data.go_to_next_page()
    self.init_user_data(user_game_data)

  def _get_score_text(self, user_data):
    game = user_data.get_game_ref()
    if game is None:
      score = 0
      time_taken = 0
    else:
      score = user_data.get_game_ref().score
      time_taken = user_data.get_game_ref().current_step

    best_score = user_data.data[Exp1UserData.USER].best_c

    text_score = "Time Taken: " + str(time_taken) + "\n"
    text_score += "Score: " + str(score) + "\n"
    text_score += "(Your Best: " + str(best_score) + ")"

    return text_score


class ToolDeliveryUserRandom(ToolDeliveryGamePage):

  def __init__(self, partial_obs) -> None:
    super().__init__(True, True, True, 5)
    self._PARTIAL_OBS = partial_obs

  def init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    user_game_data.data[Exp1UserData.SHOW_LATENT] = True
