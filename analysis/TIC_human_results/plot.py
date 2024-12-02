import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as stats
import csv
from TMM.domains.box_push_truck.simulator import BoxPushSimulatorV2
from TMM.domains.rescue.simulator import RescueSimulator

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

# MAP_SURVEY_KEY_TO_QUESTION_ID = {
#     "common_fluent": "Q1 (Fluency)",
#     "common_contributed": "Q2 (Robot Contribution)",
#     "common_improved": "Q3 (Improvement)",
#     "coach_engagement": "Q4 (Acceptance)",
#     "coach_intelligent": "Q5 (Intelligence)",
#     "coach_trust": "Q6 (Trustworthiness)",
#     "coach_effective": "Q7 (Effectiveness)",
#     "coach_timely": "Q8 (Timeliness)",
#     "coach_contributed": "Q9 (TIC Contribution)"
# }

MAP_SURVEY_KEY_TO_QUESTION_ID = {
    "common_fluent": "#1",
    "common_contributed": "#2",
    "common_improved": "#3",
    "coach_engagement": "#4",
    "coach_intelligent": "#5",
    "coach_trust": "#6",
    "coach_effective": "#7",
    "coach_timely": "#8",
    "coach_contributed": "#9"
}


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


def plot_score(ax, df, domain, use_score=True):
  column = COL_SCORE if use_score else COL_REWARD
  df_domain = df[df[COL_DOMAIN] == domain]
  df_group_a = df_domain[df_domain[COL_GROUP] == GROUPA]
  df_group_b = df_domain[df_domain[COL_GROUP] == GROUPB]
  ttest, pval = stats.ttest_ind(df_group_a[column], df_group_b[column])
  mean_a = df_group_a[column].mean()
  std_a = df_group_a[column].std()
  mean_b = df_group_b[column].mean()
  std_b = df_group_b[column].std()
  print(f"|{domain}| mean_a: {mean_a}(+-{std_a}), mean_b: {mean_b}(+-{std_b})" +
        f", t-stats: {ttest}, pval: {pval}")

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


def get_survey_summary(df, domain, group, proportion=False, keys_to_drop=None):
  df_sub = df[(df[COL_DOMAIN] == domain) & (df[COL_GROUP] == group)]
  if keys_to_drop is not None:
    df_sub = df_sub.drop(columns=keys_to_drop)
  df_sub = df_sub.dropna(axis=1)  # type: pd.DataFrame

  survey_keys = df_sub.columns[2:]
  np_counts = np.zeros((len(survey_keys), 5), dtype=int)

  for idx in range(5):
    count = df_sub[df_sub == idx].count()
    np_counts[:, idx] = count[2:]

  np_counts = (np_counts / np.sum(np_counts, axis=1, keepdims=True) *
               100 if proportion else np_counts)

  for idx, key in enumerate(survey_keys):
    midhalf = np_counts[idx, 2] / 2
    posi = np_counts[idx, 0] + np_counts[idx, 1] + midhalf
    nega = np_counts[idx, 3] + np_counts[idx, 4] + midhalf
    print(f"{domain}|{group}|{key}| Positive: {posi}, Negative: {nega}")

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
  ax.set_title(f"{MAP_SURVEY_KEY_TO_QUESTION_ID[key]}", fontsize=FONTSIZE)
  ax.set_xlabel(None)
  ax.set_xticks([])


def plot_survey_stackedbar(ax, df, domain, group, proportion=True):
  df_sub = get_survey_summary(df, domain, group, proportion=proportion)
  category_colors = plt.get_cmap('coolwarm_r')(np.linspace(
      0.15, 0.85, len(df_sub.columns)))
  ax = df_sub.plot.barh(stacked=True, ax=ax, color=category_colors)

  # X Axis
  ax.set_xlim(0, 100)

  # Y Axis
  ax.invert_yaxis()
  yticklabels = [
      MAP_SURVEY_KEY_TO_QUESTION_ID[item.get_text()]
      for item in ax.get_yticklabels()
  ]
  ax.set_yticklabels(yticklabels)


# ref: https://stackoverflow.com/a/69976552
def plot_diverging_stackedbar_impl(ax, df_sub):
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
                    height=0.8,
                    label=colname,
                    color=color)

  # Add Zero Reference Line
  ax.axvline(0, linestyle='--', color='black', alpha=.25)

  # X Axis
  ax.set_xlim(-100, 100)
  # ax.set_xticks(np.arange(-100, 101, 10))
  ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))

  # Y Axis
  ax.invert_yaxis()
  yticklabels = [
      MAP_SURVEY_KEY_TO_QUESTION_ID[item.get_text()]
      for item in ax.get_yticklabels()
  ]
  ax.set_yticks(ax.get_yticks())
  ax.set_yticklabels(yticklabels, fontsize=14)

  # Remove spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_visible(False)

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  legend_order = [0, 3, 1, 4, 2]
  ax.legend([handles[idx] for idx in legend_order],
            [labels[idx] for idx in legend_order],
            ncol=3,
            bbox_to_anchor=(0, 1),
            loc='lower left',
            fontsize='small')


# ref: https://stackoverflow.com/a/69976552
def plot_diverging_stackedbar_w_groups(ax, df_sub, n_group):
  '''
  Input Example:
                          Strongly Disagree   Disagree  ...  Strongly Agree
    common_fluent      a           6.666667  16.666667  ...        3.333333
                       b           0.000000  20.000000  ...       10.000000
    common_contributed a           6.666667  36.666667  ...        6.666667
                       b           3.333333  26.666667  ...       10.000000
    common_improved    a           6.666667  16.666667  ...       26.666667
                       b           3.333333  20.000000  ...       26.666667

  '''

  indices = list(df_sub.index)
  questions = [indices[idx][0] for idx in range(0, len(indices), n_group)]
  n_questions = len(indices) // n_group
  np_sub = df_sub.to_numpy()
  np_sub_cum = np_sub.cumsum(axis=1)
  middle_index = np_sub.shape[1] // 2
  offsets = np_sub_cum[:, middle_index - 1] + np_sub[:, middle_index] / 2

  category_names = list(df_sub.columns)
  n_cat = len(category_names)
  category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.15, 0.85, n_cat))

  hatches = ['/', None, '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
  bar_height = 0.8 / n_group
  # plot bars
  for idx in range(n_cat):
    colname = category_names[idx]
    color = category_colors[idx]
    for gidx in range(n_group):
      widths = np_sub[gidx::n_group, idx]
      starts = np_sub_cum[gidx::n_group, idx] - widths - offsets[gidx::n_group]
      rects = ax.barh(np.arange(n_questions) - bar_height * (n_group - 1) / 2 +
                      bar_height * gidx,
                      widths,
                      left=starts,
                      height=bar_height,
                      label=colname,
                      color=color,
                      hatch=hatches[gidx])

  # Add Zero Reference Line
  ax.axvline(0, linestyle='--', color='black', alpha=.25)

  # X Axis
  ax.set_xlim(-100, 100)
  # ax.set_xticks(np.arange(-100, 101, 10))
  ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))

  # Y Axis
  ax.set_yticks(np.arange(n_questions), labels=questions)
  ax.invert_yaxis()
  yticklabels = [
      MAP_SURVEY_KEY_TO_QUESTION_ID[item.get_text()]
      for item in ax.get_yticklabels()
  ]
  ax.set_yticks(ax.get_yticks())
  ax.set_yticklabels(yticklabels, fontsize=14)

  # Remove spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_visible(False)

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  legend_order = [0, 3, 1, 4, 2]
  ax.legend([handles[idx] for idx in legend_order],
            [labels[idx] for idx in legend_order],
            ncol=3,
            bbox_to_anchor=(0, 1),
            loc='lower left',
            fontsize='small')


def plot_survey_diverging_stackedbar(ax,
                                     df,
                                     domain,
                                     group,
                                     proportion=True,
                                     keys_to_drop=None):
  df_sub = get_survey_summary(df,
                              domain,
                              group,
                              proportion=proportion,
                              keys_to_drop=keys_to_drop)
  plot_diverging_stackedbar_impl(ax, df_sub)


def plot_survey_div_stackedbar_multigroup(ax, df, domain, proportion=True):
  keys_to_drop = [
      "coach_engagement", "coach_intelligent", "coach_trust", "coach_effective",
      "coach_timely", "coach_contributed"
  ]

  df_sub_a = get_survey_summary(df, domain, GROUPA, proportion, keys_to_drop)
  df_sub_b = get_survey_summary(df, domain, GROUPB, proportion, keys_to_drop)

  df_sub_a = df_sub_a.set_index(np.arange(len(df_sub_a)), append=True)
  df_sub_b = df_sub_b.set_index(np.arange(len(df_sub_b)), append=True)

  df_sub = pd.concat([df_sub_a, df_sub_b], keys=(GROUPA, GROUPB))
  df_sub = df_sub.sort_index(kind='mergesort', level=2)
  df_sub = df_sub.reset_index(level=2, drop=True)
  df_sub = df_sub.swaplevel(0, 1)

  plot_diverging_stackedbar_w_groups(ax, df_sub, 2)


def save_survey_count_plot(df, domain, group, output_dir):
  if group == GROUPA:
    fig = plt.figure(figsize=(10, 3))
    axes = fig.subplots(1, 3)
    for idx, key in enumerate(SURVEY_KEYS[:3]):
      ax = axes[idx]
      plot_survey_count(ax, df, domain, group, key)
  elif group == GROUPB:
    fig = plt.figure(figsize=(10, 8))
    axes = fig.subplots(3, 3)
    for idx, key in enumerate(SURVEY_KEYS):
      ax = axes[idx // 3, idx % 3]
      plot_survey_count(ax, df, domain, group, key)
  else:
    raise ValueError(f"Unknown group: {group}")

  title = domain[0].upper() + domain[1:]
  fig.suptitle(f"{title} (Group {group.upper()})", fontsize=16)
  fig.tight_layout()
  fig.savefig(output_dir + f"/survey_count_{domain}_{group}.png")


def save_survey_stacked_plot(df, domain, group, output_dir):
  if group == GROUPA:
    w_fig, h_fig = 7, 3.3
  elif group == GROUPB:
    w_fig, h_fig = 7, 7
  else:
    raise ValueError(f"Unknown group: {group}")

  fig = plt.figure(figsize=(w_fig, h_fig))
  ax = fig.add_subplot(1, 1, 1)
  plot_survey_stackedbar(ax, df, domain, group)

  bbox_y = -0.5 / h_fig
  bbox_x = -2 / w_fig

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  legend_order = list(range(5))
  # legend_order = [0, 3, 1, 4, 2]
  ax.legend([handles[idx] for idx in legend_order],
            [labels[idx] for idx in legend_order],
            ncol=5,
            bbox_to_anchor=(bbox_x, bbox_y),
            loc='upper left',
            fontsize='small')

  title = domain[0].upper() + domain[1:]
  fig.suptitle(f"{title} (Group {group.upper()})", fontsize=16)
  fig.tight_layout()
  fig.savefig(output_dir + f"/survey_stacked_{domain}_{group}.png")


def save_survey_divstacked_plot(df, domain, group, output_dir):
  if group == GROUPA:
    w_fig, h_fig = 7, 3
  elif group == GROUPB:
    w_fig, h_fig = 7, 7
  else:
    raise ValueError(f"Unknown group: {group}")

  fig = plt.figure(figsize=(w_fig, h_fig))
  ax = fig.add_subplot(1, 1, 1)
  plot_survey_diverging_stackedbar(ax, df, domain, group)

  bbox_y = -0.5 / h_fig
  bbox_x = -2 / w_fig

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  legend_order = list(range(5))
  # legend_order = [0, 3, 1, 4, 2]
  ax.legend([handles[idx] for idx in legend_order],
            [labels[idx] for idx in legend_order],
            ncol=5,
            bbox_to_anchor=(bbox_x, bbox_y),
            loc='upper left',
            fontsize='small')

  title = domain[0].upper() + domain[1:]
  fig.suptitle(f"{title} (Group {group.upper()})", fontsize=16)
  fig.tight_layout()
  fig.savefig(output_dir + f"/survey_divstacked_{domain}_{group}.png")


def save_survey_divstacked_plot_v2(df, output_dir):
  fig = plt.figure(figsize=(16, 4))
  axes = fig.subplots(1, 4)
  plot_survey_diverging_stackedbar(axes[0], df, MOVERS, GROUPA)
  plot_survey_diverging_stackedbar(axes[1], df, MOVERS, GROUPB)
  plot_survey_diverging_stackedbar(axes[2], df, FLOOD, GROUPA)
  plot_survey_diverging_stackedbar(axes[3], df, FLOOD, GROUPB)
  axes[0].set_xlabel("Movers (No Intervention)", fontdict={'fontsize': 14})
  axes[1].set_xlabel("Movers (TIC)", fontdict={'fontsize': 14})
  axes[2].set_xlabel("Flood (No Intervention)", fontdict={'fontsize': 14})
  axes[3].set_xlabel("Flood (TIC)", fontdict={'fontsize': 14})

  # copy axes 1 properties to axes 0
  list_yticks = axes[1].get_yticks()
  ylims = axes[1].get_ylim()
  list_yticklabels = axes[1].get_yticklabels()

  axes[0].set_ylim(*ylims)
  axes[0].set_yticks(list_yticks)
  axes[0].set_yticklabels(list_yticklabels)

  # set axes 2 ylim
  axes[2].set_ylim(*ylims)

  # remove yticks & legends
  for idx in range(1, 4):
    axes[idx].set_yticks([])
  for idx in range(4):
    axes[idx].get_legend().remove()

  # Legend
  handles, labels = axes[0].get_legend_handles_labels()
  legend_order = list(range(5))
  fig.legend([handles[idx] for idx in legend_order],
             [labels[idx] for idx in legend_order],
             ncol=5,
             bbox_to_anchor=(0.14, 0.98),
             loc='lower left',
             fontsize='medium')

  fig.tight_layout()
  fig.savefig(output_dir + "/survey_divstacked_all.png", bbox_inches='tight')


def save_survey_divstacked_plot_v3(df, output_dir, is_common):
  if is_common:
    fig = plt.figure(figsize=(9, 4))
    axes = fig.subplots(1, 2)
    plot_survey_div_stackedbar_multigroup(axes[0], df, MOVERS)
    plot_survey_div_stackedbar_multigroup(axes[1], df, FLOOD)

    q_type = "common"
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles[1::2]
    labels = labels[1::2]
    labels = [txt.replace(" ", "\n") for txt in labels]
    group_handles = [
        Patch(facecolor='white', edgecolor='black', hatch='//'),
        Patch(facecolor='white', edgecolor='black', hatch=None)
    ]
    group_labels = ["No Intervention", "TIC"]
    final_handles = handles + group_handles
    final_labels = labels + group_labels
    legend_order = [0, 5, 1, 6, 2, 3, 4]
    fig.legend([final_handles[idx] for idx in legend_order],
               [final_labels[idx] for idx in legend_order],
               ncol=5,
               bbox_to_anchor=(0.02, 0.98),
               loc='lower left',
               fontsize='large')

  else:
    fig = plt.figure(figsize=(9, 4))
    axes = fig.subplots(1, 2)
    keys_to_drop = ["common_fluent", "common_contributed", "common_improved"]
    plot_survey_diverging_stackedbar(axes[0],
                                     df,
                                     MOVERS,
                                     GROUPB,
                                     keys_to_drop=keys_to_drop)
    plot_survey_diverging_stackedbar(axes[1],
                                     df,
                                     FLOOD,
                                     GROUPB,
                                     keys_to_drop=keys_to_drop)
    q_type = "coach"
    NO_LEGEND = False
    if NO_LEGEND:
      # Legend
      handles, labels = axes[0].get_legend_handles_labels()
      labels = [txt.replace(" ", "\n") for txt in labels]
      legend_order = list(range(5))
      fig.legend([handles[idx] for idx in legend_order],
                 [labels[idx] for idx in legend_order],
                 ncol=5,
                 bbox_to_anchor=(0.02, 0.98),
                 loc='lower left',
                 fontsize='large')

  axes[0].set_xlabel("Movers", fontdict={'fontsize': 14})
  axes[1].set_xlabel("Flood", fontdict={'fontsize': 14})

  # remove yticks & legends
  axes[1].set_yticks([])
  for idx in range(2):
    axes[idx].get_legend().remove()

  fig.tight_layout()

  fig.savefig(output_dir + f"/survey_divstacked_{q_type}.png",
              bbox_inches='tight')


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
  data_dir = os.path.join(cur_dir, "data")
  survey_dir = os.path.join(data_dir, "tw2020_survey")
  traj_dir = os.path.join(data_dir, "tw2020_trajectory")
  intv_dir = os.path.join(data_dir, "tw2020_user_label")
  output_dir = os.path.join(cur_dir, "output")

  PLOT_TASK_RESULTS = False
  PLOT_SURVEY_COUNT = False
  PLOT_SURVEY_STACKED = False
  PLOT_SURVEY_DIVSTACKED = False
  PLOT_SURVEY_DIVSTACKED2 = False
  PLOT_SURVEY_DIVSTACKED3 = True
  if PLOT_TASK_RESULTS:
    USE_SCORE = True
    df = conv_task_results_2_df(traj_dir, intv_dir)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot_score(ax1, df, MOVERS, use_score=USE_SCORE)
    plot_score(ax2, df, FLOOD, use_score=USE_SCORE)

    stat_num_intervention(df, MOVERS)
    stat_num_intervention(df, FLOOD)

    fig.tight_layout()
    txt_score = "score" if USE_SCORE else "reward"
    fig.savefig(output_dir + f"/task_results_{txt_score}.png")
    plt.show()

  if PLOT_SURVEY_COUNT:
    df = conv_survey_2_df(survey_dir)
    save_survey_count_plot(df, MOVERS, GROUPB, output_dir)
    save_survey_count_plot(df, FLOOD, GROUPB, output_dir)
    save_survey_count_plot(df, MOVERS, GROUPA, output_dir)
    save_survey_count_plot(df, FLOOD, GROUPA, output_dir)
    plt.show()

  if PLOT_SURVEY_STACKED:
    df = conv_survey_2_df(survey_dir)
    save_survey_stacked_plot(df, MOVERS, GROUPB, output_dir)
    save_survey_stacked_plot(df, FLOOD, GROUPB, output_dir)
    save_survey_stacked_plot(df, MOVERS, GROUPA, output_dir)
    save_survey_stacked_plot(df, FLOOD, GROUPA, output_dir)
    plt.show()

  if PLOT_SURVEY_DIVSTACKED:
    df = conv_survey_2_df(survey_dir)

    save_survey_divstacked_plot(df, MOVERS, GROUPB, output_dir)
    save_survey_divstacked_plot(df, FLOOD, GROUPB, output_dir)
    save_survey_divstacked_plot(df, MOVERS, GROUPA, output_dir)
    save_survey_divstacked_plot(df, FLOOD, GROUPA, output_dir)
    plt.show()

  if PLOT_SURVEY_DIVSTACKED2:
    df = conv_survey_2_df(survey_dir)
    save_survey_divstacked_plot_v2(df, output_dir)
    plt.show()

  if PLOT_SURVEY_DIVSTACKED3:
    df = conv_survey_2_df(survey_dir)
    save_survey_divstacked_plot_v3(df, output_dir, is_common=True)
    save_survey_divstacked_plot_v3(df, output_dir, is_common=False)
    plt.show()
