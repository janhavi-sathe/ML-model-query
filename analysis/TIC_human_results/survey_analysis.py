import os
import numpy as np
from plot import (conv_survey_2_df, MOVERS, FLOOD, get_survey_summary, GROUPB,
                  GROUPA, COL_DOMAIN, COL_GROUP)
import scipy.stats as stats

COMMON_KEYS = ["common_fluent", "common_contributed", "common_improved"]


def get_common_survey_stats(df, domain, column):
  df_sub_a = df[(df[COL_DOMAIN] == domain) & (df[COL_GROUP] == GROUPA)]
  df_sub_b = df[(df[COL_DOMAIN] == domain) & (df[COL_GROUP] == GROUPB)]

  ttest, pval = stats.ttest_ind(df_sub_a[column], df_sub_b[column])
  mean_a = df_sub_a[column].mean()
  std_a = df_sub_a[column].std()
  mean_b = df_sub_b[column].mean()
  std_b = df_sub_b[column].std()
  print(
      f"|{domain}| " +
      f"Control: {mean_a:.2f}(+-{std_a:.2f}), Exp.: {mean_b:.2f}(+-{std_b:.2f})"
      + f", p-val: {pval:.2f}")


def get_aicoach_survey_stats(df, domain):
  df_sub = get_survey_summary(df,
                              domain,
                              GROUPB,
                              proportion=True,
                              keys_to_drop=COMMON_KEYS)

  df_sum = df_sub.sum(axis=0) / len(df_sub) / 100
  mean_all = (df_sum["Strongly Disagree"] * 1 + df_sum["Disagree"] * 2 +
              df_sum["Neutral"] * 3 + df_sum["Agree"] * 4 +
              df_sum["Strongly Agree"] * 5)

  var_all = ((1 - mean_all)**2 * df_sum["Strongly Disagree"] +
             (2 - mean_all)**2 * df_sum["Disagree"] +
             (3 - mean_all)**2 * df_sum["Neutral"] +
             (4 - mean_all)**2 * df_sum["Agree"] +
             (5 - mean_all)**2 * df_sum["Strongly Agree"])
  # std
  std_all = np.sqrt(var_all)

  print(mean_all, "+-", std_all)


if __name__ == "__main__":

  cur_dir = os.path.dirname(__file__)
  data_dir = os.path.join(cur_dir, "data")
  survey_dir = os.path.join(data_dir, "tw2020_survey")
  traj_dir = os.path.join(data_dir, "tw2020_trajectory")
  intv_dir = os.path.join(data_dir, "tw2020_user_label")
  output_dir = os.path.join(cur_dir, "output")

  df = conv_survey_2_df(survey_dir)

  get_common_survey_stats(df, MOVERS, COMMON_KEYS[0])
  get_common_survey_stats(df, MOVERS, COMMON_KEYS[1])
  get_common_survey_stats(df, MOVERS, COMMON_KEYS[2])
  get_common_survey_stats(df, FLOOD, COMMON_KEYS[0])
  get_common_survey_stats(df, FLOOD, COMMON_KEYS[1])
  get_common_survey_stats(df, FLOOD, COMMON_KEYS[2])

  get_aicoach_survey_stats(df, MOVERS)
  get_aicoach_survey_stats(df, FLOOD)
