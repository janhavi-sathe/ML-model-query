import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
from aic_domain.rescue.simulator import RescueSimulator

sns.set_theme(style="white")

# consts
COL_DOMAIN = 'domain'
COL_GROUP = 'group'
COL_REWARD = 'reward'
COL_NUM_INTV = 'num_intervention'
COL_SCORE = 'score'
MOVERS = 'movers'
FLOOD = 'flood'
GROUPA = 'a'
GROUPB = 'b'
SURVEY_KEYS = [
    "common_fluent", "common_contributed", "common_improved",
    "coach_engagement", "coach_intelligent", "coach_trust", "coach_effective",
    "coach_timely", "coach_contributed"
]


def get_user_names(dir_path):
  group_a = []
  group_b = []
  user_dirs = glob.glob(os.path.join(dir_path, "*"))

  for user_dir in user_dirs:
    user_name = os.path.basename(user_dir)
    if user_name.startswith("a"):
      group_a.append(user_name)
    elif user_name.startswith("b"):
      group_b.append(user_name)
    else:
      raise ValueError(f"Unknown user name: {user_name}")

  return group_a, group_b


def conv_task_results_2_df(traj_dir, intv_dir, movers_cost=1):
  group_a, group_b = get_user_names(traj_dir)

  rows = []
  TRAJ_PREFIX = "ntrv_session_"
  INTV_PREFIX = "interventions_ntrv_session_"
  for idx in range(2, 5):
    # NOTE: for group A, score = reward as no intervention
    for name in group_a:
      # movers
      tname = glob.glob(os.path.join(traj_dir, name,
                                     TRAJ_PREFIX + f"a{idx}*"))[0]
      traj = BoxPushSimulatorV2.read_file(tname)
      reward = 150 - len(traj)
      rows.append((MOVERS, GROUPA, reward, None, reward))
      # flood
      tname = glob.glob(os.path.join(traj_dir, name,
                                     TRAJ_PREFIX + f"c{idx}*"))[0]
      traj = RescueSimulator.read_file(tname)
      reward = traj[-1][0]
      rows.append((FLOOD, GROUPA, reward, None, reward))

    for name in group_b:
      # movers
      tname = glob.glob(os.path.join(traj_dir, name,
                                     TRAJ_PREFIX + f"a{idx}*"))[0]
      iname = glob.glob(os.path.join(intv_dir, name,
                                     INTV_PREFIX + f"a{idx}*"))[0]
      traj = BoxPushSimulatorV2.read_file(tname)
      reward = 150 - len(traj)
      with open(iname, newline='') as txtfile:
        n_intv = len(txtfile.readlines()) - 1
      score = reward - movers_cost * n_intv  # incorporate intervention cost
      rows.append((MOVERS, GROUPB, reward, n_intv, score))
      # flood (intervention cost = 0)
      tname = glob.glob(os.path.join(traj_dir, name,
                                     TRAJ_PREFIX + f"c{idx}*"))[0]
      iname = glob.glob(os.path.join(intv_dir, name,
                                     INTV_PREFIX + f"c{idx}*"))[0]
      traj = RescueSimulator.read_file(tname)
      reward = traj[-1][0]
      with open(iname, newline='') as txtfile:
        n_intv = len(txtfile.readlines()) - 1
      rows.append((FLOOD, GROUPB, reward, n_intv, reward))

  df = pd.DataFrame(
      rows,
      columns=[COL_DOMAIN, COL_GROUP, COL_REWARD, COL_NUM_INTV, COL_SCORE])
  return df

  # df indices: (domain, group)
  # df columns: performance, # intervention


def conv_survey_2_df(survey_dir):
  rows = []
  participants = glob.glob(os.path.join(survey_dir, "*"))
  for dir in participants:
    uname = os.path.basename(dir)
    if uname.startswith("a"):
      ugroup = GROUPA
    elif uname.startswith("b"):
      ugroup = GROUPB
    else:
      raise ValueError(f"User group can't be inferred from user name: {uname}")

    sname = glob.glob(os.path.join(dir, "insurvey*"))[0]
    with open(sname, mode='r') as file:
      dict_csv = csv.DictReader(file)
      count = 0
      for csv_row in dict_csv:
        count += 1
        if csv_row['session'] == 'ntrv_session_a4':
          domain = MOVERS
        elif csv_row['session'] == 'ntrv_session_c4':
          domain = FLOOD
        else:
          raise ValueError(f"Unknown session: {csv_row['session']}")

        list_values = []
        for key in SURVEY_KEYS:
          if csv_row[key] == 'N/A':
            list_values.append(None)
          else:
            list_values.append(int(csv_row[key]))

        rows.append((domain, ugroup, *list_values))

  df = pd.DataFrame(rows, columns=[COL_DOMAIN, COL_GROUP, *SURVEY_KEYS])

  return df


def plot_score(ax, df, domain):
  column = COL_SCORE
  df_domain = df[df[COL_DOMAIN] == domain]
  df_group_a = df_domain[df_domain[COL_GROUP] == GROUPA]
  df_group_b = df_domain[df_domain[COL_GROUP] == GROUPB]
  ttest, pval = stats.ttest_ind(df_group_a[column], df_group_b[column])
  print(f"|{domain}| t-stats: {ttest}, pval: {pval}")

  ax = sns.boxplot(ax=ax,
                   data=df_domain,
                   x=COL_GROUP,
                   y=column,
                   medianprops={
                       'color': "firebrick",
                       'linewidth': 2,
                       'linestyle': '--'
                   },
                   showmeans=True)

  FONTSIZE = 14

  title = domain[0].upper() + domain[1:]
  ax.set_title(f"{title}", fontsize=FONTSIZE)
  ax.set_ylabel("Score", fontsize=FONTSIZE)
  ax.set_xlabel(None)
  ax.set_xticklabels(['No Intervention', 'TIC'], fontsize=FONTSIZE)


def stat_num_intervention(df, domain):
  column = COL_NUM_INTV
  df_domain_group_b = df[(df[COL_DOMAIN] == domain) & (df[COL_GROUP] == GROUPB)]
  print(f"===== {domain} =====")
  print(df_domain_group_b[column].describe())


def get_survey_summary(df, domain, group, proportion=False):
  df_sub = df[(df[COL_DOMAIN] == domain) & (df[COL_GROUP] == group)]
  df_sub = df_sub.dropna(axis=1)  # type: pd.DataFrame

  survey_keys = df_sub.columns[2:]
  np_counts = np.zeros((len(survey_keys), 5), dtype=int)

  for idx in range(5):
    count = df_sub[df_sub == idx].count()
    np_counts[:, idx] = count[2:]

  np_counts = (np_counts / np.sum(np_counts, axis=1, keepdims=True) *
               100 if proportion else np_counts)

  df_counts = pd.DataFrame(np_counts,
                           columns=[
                               "Strongly Agree", "Agree", "Neutral", "Disagree",
                               "Strongly Disagree"
                           ],
                           index=survey_keys)

  # change order of categories
  cat_order = [
      "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"
  ]
  df_counts = df_counts[cat_order]
  return df_counts


def plot_survey_count(ax, df, domain, group, key):
  df_sub = get_survey_summary(df, domain, group, proportion=False)
  df_sub_tidyform = df_sub.reset_index().melt('index',
                                              var_name='category',
                                              value_name='counts')
  df_key = df_sub_tidyform[df_sub_tidyform['index'] == key]

  cat_order = [
      "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"
  ]
  category_colors = plt.get_cmap('coolwarm_r')(np.linspace(
      0.15, 0.85, len(cat_order)))
  # NOTE: to add legend, set hue='category' in sns.barplot
  ax = sns.barplot(ax=ax,
                   data=df_key,
                   y='counts',
                   x='category',
                   palette=category_colors,
                   order=cat_order)

  FONTSIZE = 12
  ax.set_title(f"{key}", fontsize=FONTSIZE)
  ax.set_xlabel(None)
  ax.set_xticks([])


def plot_survey_stackedbar(ax, df, domain, group, proportion=True):
  df_sub = get_survey_summary(df, domain, group, proportion=proportion)
  category_colors = plt.get_cmap('coolwarm_r')(np.linspace(
      0.15, 0.85, len(df_sub.columns)))
  df_sub.plot.barh(stacked=True, ax=ax, color=category_colors)


# ref: https://stackoverflow.com/a/69976552
def plot_survey_diverging_stackedbar(ax, df, domain, group, proportion=True):
  df_sub = get_survey_summary(df, domain, group, proportion=proportion)
  questions = list(df_sub.index)
  np_sub = df_sub.to_numpy()
  np_sub_cum = np_sub.cumsum(axis=1)
  middle_index = np_sub.shape[1] // 2
  offsets = np_sub_cum[:, middle_index - 1] + np_sub[:, middle_index] / 2

  category_names = list(df_sub.columns)
  n_cat = len(category_names)
  category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.15, 0.85, n_cat))

  # plot bars
  for idx in range(n_cat):
    colname = category_names[idx]
    color = category_colors[idx]
    widths = np_sub[:, idx]
    starts = np_sub_cum[:, idx] - widths - offsets
    rects = ax.barh(questions,
                    widths,
                    left=starts,
                    height=0.5,
                    label=colname,
                    color=color)

  # Add Zero Reference Line
  ax.axvline(0, linestyle='--', color='black', alpha=.25)

  # X Axis
  ax.set_xlim(-100, 100)
  ax.set_xticks(np.arange(-100, 101, 10))
  ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))

  # Y Axis
  ax.invert_yaxis()

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  # dict_legend = dict(zip(labels, handles))
  legend_order = [0, 3, 1, 4, 2]
  ax.legend([handles[idx] for idx in legend_order],
            [labels[idx] for idx in legend_order],
            ncol=3,
            bbox_to_anchor=(0, 1),
            loc='lower left',
            fontsize='small')


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
  data_dir = os.path.join(cur_dir, "data")
  survey_dir = os.path.join(data_dir, "tw2020_survey")
  traj_dir = os.path.join(data_dir, "tw2020_trajectory")
  intv_dir = os.path.join(data_dir, "tw2020_user_label")

  PLOT_TASK_RESULTS = False
  PLOT_SURVEY_COUNT = False
  PLOT_SURVEY_STACKED = False
  PLOT_SURVEY_DIVSTACKED = True
  if PLOT_TASK_RESULTS:
    df = conv_task_results_2_df(traj_dir, intv_dir)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot_score(ax1, df, MOVERS)
    plot_score(ax2, df, FLOOD)

    stat_num_intervention(df, MOVERS)
    stat_num_intervention(df, FLOOD)

    fig.tight_layout()
    plt.show()

  if PLOT_SURVEY_COUNT:
    df = conv_survey_2_df(survey_dir)

    # movers group b
    fig = plt.figure(figsize=(10, 8))
    axes = fig.subplots(3, 3)
    for idx, key in enumerate(SURVEY_KEYS):
      ax = axes[idx // 3, idx % 3]
      plot_survey_count(ax, df, MOVERS, GROUPB, key)
    fig.suptitle("Movers Group B", fontsize=16)
    fig.tight_layout()

    # flood group b
    fig = plt.figure(figsize=(10, 8))
    axes = fig.subplots(3, 3)
    for idx, key in enumerate(SURVEY_KEYS):
      ax = axes[idx // 3, idx % 3]
      plot_survey_count(ax, df, FLOOD, GROUPB, key)
    fig.suptitle("Flood Group B", fontsize=16)
    fig.tight_layout()

    # movers group a
    fig = plt.figure(figsize=(10, 3))
    axes = fig.subplots(1, 3)
    for idx, key in enumerate(SURVEY_KEYS[:3]):
      ax = axes[idx]
      plot_survey_count(ax, df, MOVERS, GROUPA, key)
    fig.suptitle("Movers Group A", fontsize=16)
    fig.tight_layout()

    # flood group a
    fig = plt.figure(figsize=(10, 3))
    axes = fig.subplots(1, 3)
    for idx, key in enumerate(SURVEY_KEYS[:3]):
      ax = axes[idx]
      plot_survey_count(ax, df, FLOOD, GROUPA, key)
    fig.suptitle("Flood Group A", fontsize=16)
    fig.tight_layout()

    plt.show()

  if PLOT_SURVEY_STACKED:
    df = conv_survey_2_df(survey_dir)

    fig = plt.figure(figsize=(6, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_stackedbar(ax1, df, MOVERS, GROUPB)
    fig.suptitle("Movers Group B", fontsize=16)
    fig.tight_layout()

    fig = plt.figure(figsize=(6, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_stackedbar(ax1, df, FLOOD, GROUPB)
    fig.suptitle("Flood Group B", fontsize=16)
    fig.tight_layout()

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_stackedbar(ax1, df, MOVERS, GROUPA)
    fig.suptitle("Movers Group A", fontsize=16)
    fig.tight_layout()

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_stackedbar(ax1, df, FLOOD, GROUPA)
    fig.suptitle("Flood Group A", fontsize=16)
    fig.tight_layout()

    plt.show()

  if PLOT_SURVEY_DIVSTACKED:
    df = conv_survey_2_df(survey_dir)

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_diverging_stackedbar(ax1, df, MOVERS, GROUPB)
    fig.suptitle("Movers Group B", fontsize=16)
    fig.tight_layout()

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_diverging_stackedbar(ax1, df, FLOOD, GROUPB)
    fig.suptitle("Flood Group B", fontsize=16)
    fig.tight_layout()

    fig = plt.figure(figsize=(7, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_diverging_stackedbar(ax1, df, MOVERS, GROUPA)
    fig.suptitle("Movers Group A", fontsize=16)
    fig.tight_layout()

    fig = plt.figure(figsize=(7, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    plot_survey_diverging_stackedbar(ax1, df, FLOOD, GROUPA)
    fig.suptitle("Flood Group A", fontsize=16)
    fig.tight_layout()

    plt.show()
