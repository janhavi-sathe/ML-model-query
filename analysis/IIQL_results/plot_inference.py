import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

IDIL = "IDIL"
IDIL_J = "IDIL-J"
OGAIL = "Option-GAIL"

MG3 = "MG-3"
MG5 = "MG-5"
ONEMOVER = "OneMover"
MOVERS = "Movers"

COL_ACC = "Accuracy"
COL_MET = "Method"
COL_DOM = "Domain"
COLUMNS = [COL_DOM, COL_MET, COL_ACC]

rows = []
rows.append((MG3, OGAIL, 0.3911))
rows.append((MG3, OGAIL, 0.3756))
rows.append((MG3, OGAIL, 0.5489))
rows.append((MG3, IDIL_J, 0.6005))
rows.append((MG3, IDIL_J, 0.5521))
rows.append((MG3, IDIL_J, 0.7781))
rows.append((MG3, IDIL, 0.9503))
rows.append((MG3, IDIL, 0.9158))
rows.append((MG3, IDIL, 0.9275))
rows.append((MG5, OGAIL, 0.2067))
rows.append((MG5, OGAIL, 0.1804))
rows.append((MG5, OGAIL, 0.2069))
rows.append((MG5, IDIL_J, 0.4224))
rows.append((MG5, IDIL_J, 0.7813))
rows.append((MG5, IDIL_J, 0.3889))
rows.append((MG5, IDIL, 0.7225))
rows.append((MG5, IDIL, 0.8675))
rows.append((MG5, IDIL, 0.8924))
rows.append((ONEMOVER, OGAIL, 0.3172))
rows.append((ONEMOVER, OGAIL, 0.4748))
rows.append((ONEMOVER, OGAIL, 0.3502))
rows.append((ONEMOVER, IDIL_J, 0.4773))
rows.append((ONEMOVER, IDIL_J, 0.3333))
rows.append((ONEMOVER, IDIL_J, 0.1555))
rows.append((ONEMOVER, IDIL, 0.5738))
rows.append((ONEMOVER, IDIL, 0.6597))
rows.append((ONEMOVER, IDIL, 0.6271))
rows.append((MOVERS, OGAIL, 0.5016))
rows.append((MOVERS, OGAIL, 0.0922))
rows.append((MOVERS, OGAIL, 0.5589))
rows.append((MOVERS, IDIL_J, 0.5174))
rows.append((MOVERS, IDIL_J, 0.5556))
rows.append((MOVERS, IDIL_J, 0.3687))
rows.append((MOVERS, IDIL, 0.6623))
rows.append((MOVERS, IDIL, 0.5109))
rows.append((MOVERS, IDIL, 0.6215))

df = pd.DataFrame(rows, columns=COLUMNS)

fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(111)
ax = sns.barplot(ax=ax, data=df, x=COL_DOM, y=COL_ACC, hue=COL_MET)

h_len = 0.25
margin = h_len * 0.05
idx = 0
ax.axhline(y=.33,
           xmin=h_len * idx + margin,
           xmax=h_len * (idx + 1) - margin,
           color='dimgray',
           linestyle='--')
idx += 1
ax.axhline(y=.20,
           xmin=h_len * idx + margin,
           xmax=h_len * (idx + 1) - margin,
           color='dimgray',
           linestyle='--')
idx += 1
ax.axhline(y=.25,
           xmin=h_len * idx + margin,
           xmax=h_len * (idx + 1) - margin,
           color='dimgray',
           linestyle='--')
idx += 1
ax.axhline(y=.20,
           xmin=h_len * idx + margin,
           xmax=h_len * (idx + 1) - margin,
           color='dimgray',
           linestyle='--')

ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))

fig.tight_layout()
plt.show()
