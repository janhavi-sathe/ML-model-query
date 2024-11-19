import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="white")

BC = "BC"
IQL = "IQLearn"
IDIL = "IDIL"
IDIL_J = "IDIL-J"
OGAIL = "Option-GAIL"
IDIL_S = "IDIL-s"
IDIL_J_S = "IDIL-J-s"
OGAIL_S = "Option-GAIL-s"

MG3 = "MG-3"
MG5 = "MG-5"
ONEMOVER = "OneMover"
MOVERS = "Movers"

HOPPER = "Hopper"
HALFCHEETAH = "HalfCheetah"
WALKER = "Walker"
ANT = "Ant"
HUMANOID = "Humanoid"
ANTPUSH = "AntPush"

COL_REW = "Reward"
COL_MET = "Method"
COL_DOM = "Domain"
COLUMNS = [COL_DOM, COL_MET, COL_REW]

MET_GRP_1 = [BC, IQL, OGAIL, IDIL_J, IDIL]
MET_GRP_2 = [BC, IQL, OGAIL, IDIL_J, IDIL, OGAIL_S, IDIL_J_S, IDIL_S]

EXPERT = {
    MG3: 22.32,
    MG5: 40.09,
    ONEMOVER: -46.8,
    MOVERS: -67.6,
    HOPPER: 3533,
    HALFCHEETAH: 5098,
    WALKER: 5274,
    ANT: 4700,
    HUMANOID: 5313,
    ANTPUSH: 116.6
}

rows = []

# Hopper
np_data = np.array([[930, 3541, 2790, 3574,
                     3508], [1381, 3530, 3571, 3587, 3537],
                    [271, 3499, 3477, 3512, 3524]])
for row in range(3):
  for col, method in enumerate(MET_GRP_1):
    rows.append((HOPPER, method, np_data[row][col]))

# HalfCheetah
np_data = np.array([[3957, 5074, 1787, 5178, 5128],
                    [4509, 5074, 3024, 5125, 5143],
                    [4493, 5129, 3593, 5106, 5022]])
for row in range(3):
  for col, method in enumerate(MET_GRP_1):
    rows.append((HALFCHEETAH, method, np_data[row][col]))

# Walker
np_data = np.array([[3081, 5139, 5043, 5200, 5218],
                    [2258, 5224, 2321, 5143, 5262],
                    [3857, 5191, 5171, 5195, 5235]])
for row in range(3):
  for col, method in enumerate(MET_GRP_1):
    rows.append((WALKER, method, np_data[row][col]))

# Ant
np_data = np.array([[4353, 4571, 4516, 4620, 4604],
                    [4002, 4647, 4594, 4621, 4589],
                    [4183, 4601, 4479, 4667, 4547]])
for row in range(3):
  for col, method in enumerate(MET_GRP_1):
    rows.append((ANT, method, np_data[row][col]))

# Humanoid
np_data = np.array([[528, 5315, 508, 5274, 5443], [443, 5417, 439, 5315, 5427],
                    [518, 5318, 429, 5307, 5240]])
for row in range(3):
  for col, method in enumerate(MET_GRP_1):
    rows.append((HUMANOID, method, np_data[row][col]))

# Antpush
np_data = np.array([[96.9, 117.3, 108.7, 118.0, 117.0],
                    [74.7, 116.7, 80.2, 114.2, 116.9],
                    [98.9, 115.8, 107.6, 117.7, 117.3]])
for row in range(3):
  for col, method in enumerate(MET_GRP_1):
    rows.append((ANTPUSH, method, np_data[row][col]))

# MG-3
np_data = np.array([[-8.75, 3.87, -10.00, -3.75, 22.38, 1.58, -3.75, 22.49],
                    [-6.24, -7.50, -10.00, -3.75, 18.16, -6.25, -2.50, 21.35],
                    [-6.25, 21.80, -13.75, -6.25, 21.38, -10.00, -3.75, 22.93]])
for row in range(3):
  for col, method in enumerate(MET_GRP_2):
    rows.append((MG3, method, np_data[row][col]))

# MG-5
np_data = np.array([[-5.00, -9.00, -10.00, 0.00, 26.43, -6.25, 6.25, 33.65],
                    [-6.26, -11.25, -13.75, 3.75, 27.35, -15.00, 8.80, 32.63],
                    [-7.50, -12.50, -15.00, 8.75, 29.88, -15.00, 3.75, 32.25]])
for row in range(3):
  for col, method in enumerate(MET_GRP_2):
    rows.append((MG5, method, np_data[row][col]))

# OneMover
np_data = np.array(
    [[-200.0, -42.9, -200.0, -71.5, -44.8, -200.0, -53.9, -43.9],
     [-200.0, -47.6, -200.0, -97.9, -46.625, -200.0, -106.9, -46.8],
     [-200.0, -46.5, -200.0, -104.4, -45.0, -200.0, -51.0, -47.5]])
for row in range(3):
  for col, method in enumerate(MET_GRP_2):
    rows.append((ONEMOVER, method, np_data[row][col]))

# Movers
np_data = np.array(
    [[-200.0, -65.3, -200.0, -162.7, -70.5, -200.0, -179.7, -68.4],
     [-200.0, -66.1, -200.0, -134.0, -65.5, -200.0, -160.7, -68.5],
     [-200.0, -65.4, -200.0, -160.2, -68.9, -200.0, -125.2, -67.6]])
for row in range(3):
  for col, method in enumerate(MET_GRP_2):
    rows.append((MOVERS, method, np_data[row][col]))

df = pd.DataFrame(rows, columns=COLUMNS)

df = df.groupby([COL_DOM, COL_MET])[COL_REW].mean().reset_index()

fig = plt.figure(figsize=(16, 2.5))
axes = fig.subplots(1, 6)
for idx, dom in enumerate([HOPPER, HALFCHEETAH, WALKER, ANT, HUMANOID,
                           ANTPUSH]):
  ax = sns.pointplot(ax=axes[idx],
                     data=df[df[COL_DOM] == dom],
                     x=COL_MET,
                     y=COL_REW,
                     hue=COL_MET,
                     order=MET_GRP_1,
                     hue_order=MET_GRP_1)
  # ax.tick_params(axis='y', which='major', pad=-5.5)
  ax.set_ylim(bottom=0, top=None)
  ax.set_ylabel("")
  ax.set_xticklabels([])
  ax.set_xlabel(dom, fontsize=14)
  ax.get_legend().remove()

  ax.axhline(y=EXPERT[dom], color='dimgray', linestyle='--')

fig.tight_layout()
# Legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].set_ylabel(COL_REW, fontsize=14)
axes[0].legend(handles,
               labels,
               ncol=6,
               bbox_to_anchor=(0, 1),
               loc='lower left',
               fontsize=14)

cur_dir = os.path.dirname(__file__)
fig.savefig(cur_dir + "/plot_mjc.png", bbox_inches='tight')

# ----------------------------------------------------------------------------
fig2 = plt.figure(figsize=(16, 2.5))
axes2 = fig2.subplots(1, 4)
for idx, dom in enumerate([MG3, MG5, ONEMOVER, MOVERS]):
  ax = sns.pointplot(ax=axes2[idx],
                     data=df[df[COL_DOM] == dom],
                     x=COL_MET,
                     y=COL_REW,
                     hue=COL_MET,
                     order=MET_GRP_2,
                     hue_order=MET_GRP_2)
  # ax.tick_params(axis='y', which='major', pad=-5.5)
  ax.set_ylabel("")
  ax.set_xticklabels([])
  ax.set_xlabel(dom, fontsize=14)
  ax.get_legend().remove()

  ax.axhline(y=EXPERT[dom], color='dimgray', linestyle='--')

axes2[0].set_ylim(bottom=-20, top=None)
axes2[1].set_ylim(bottom=-20, top=None)

fig2.tight_layout()
# Legend
handles, labels = axes2[0].get_legend_handles_labels()
axes2[0].set_ylabel(COL_REW, fontsize=14)
axes2[0].legend(handles,
                labels,
                ncol=9,
                bbox_to_anchor=(0, 1),
                loc='lower left',
                fontsize=14)

cur_dir = os.path.dirname(__file__)
fig2.savefig(cur_dir + "/plot_intent.png", bbox_inches='tight')

plt.show()
