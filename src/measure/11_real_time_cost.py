# -*- coding:utf8 -*-

# import pylab
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# import numpy as np

# v=3
# fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
# inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
# fig_width = fig_width_pt*inches_per_pt  # width in inches
# # fig_height = fig_width*golden_mean      # height in inches
# fig_height = fig_width*0.8
# fig_size =  [fig_width*v,fig_height*v]
# params = {'backend': 'ps',
#           'axes.labelsize': 30,
#           'text.fontsize': 30,
#           'legend.fontsize': 30,
#           'xtick.labelsize': 30,
#           'ytick.labelsize': 30,
#           'lines.linewidth': 4,
#           'text.usetex': True,
#           'font.family':'Times New Roman',
#           'figure.figsize': fig_size}
# pylab.rcParams.update(params)

# w2d = [[5.10E-05, 5.10E-02,  5.10E-01,  5.10E+00],
# [3.07E+01,  3.07E+04,  3.07E+05,  3.07E+05],
# [28.78762031,    2.88E+04,  2.88E+05,  2.88E+06],
# [2143.356982,    2.14E+06,  2.14E+07,  2.14E+08],
# [14112.59065,    1.41E+07,  1.41E+08,  1.41E+09]]

# d2w = [[4.02E+01,  4.02E+04,  4.02E+05,  4.02E+06],
# [1.18E+01,  1.18E+04,  1.18E+05,  1.18E+05],
# [11.09192753,    1.11E+04,  1.11E+05,  1.11E+06],
# [825.8397207,    8.26E+05,  8.26E+06,  8.26E+07],
# [5437.609329,    5.44E+06,  5.44E+07,  5.44E+08]]

# syn = [[5.10216E-05,   0.005102158,    5.102157593,    51021.57593],
# [0.15489006,     15.48900604,    15489.00604,    154890060.4],
# [1.45E-01,  1.45E+01,  1.45E+04,  1.45E+08],
# [1.08E+01,  1.08E+03,  1.08E+06,  1.08E+10],
# [71.11086968,    7111.086968,    71110.86968,    711108.6968]]

# methods = ['OURS', 'PALE-MLP', 'IONE', 'MNA', 'FRUI-P']


# w2d = np.array(w2d)
# d2w = np.array(d2w)
# syn = np.array(syn)

# fig, ax = plt.subplots()

# for i in range(len(methods)):
#      ax.bar(w2d[i,:], [1, 1000, 10000, 100000])

# ax.legend(loc=5)

# plt.show()

import pylab
import matplotlib.pyplot as plt
import numpy as np

v=1.7
fig_width_pt = 350.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
# fig_height = fig_width*golden_mean      # height in inches
fig_height = fig_width*0.65
fig_size =  [fig_width*v,fig_height*v]
params = {'axes.labelsize': 12,
          'font.size': 14,
          'legend.fontsize': 15,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'text.usetex': True,
          'font.family':'Times New Roman',
          'figure.figsize': fig_size}
pylab.rcParams.update(params)

w2d = [[5.10E-03, 5.10E-02,  5.10E-01,  5.10E+00],
[3.07E+03,  3.07E+04,  3.07E+05,  3.07E+06],
[2878.762031,    2.88E+04,  2.88E+05,  2.88E+06],
[214335.6982,    2.14E+06,  2.14E+07,  2.14E+08],
[1411259.065,    1.41E+07,  1.41E+08,  1.41E+09]]

d2w = [[5.10E-03,    5.10E-02, 5.10E-01,  5.10E+00],
[1.18E+03,  1.18E+04,  1.18E+05,  1.18E+06],
[1109.192753,    1.11E+04,  1.11E+05,  1.11E+06],
[82583.97207,    8.26E+05,  8.26E+06,  8.26E+07],
[543760.9329,    5.44E+06,  5.44E+07,  5.44E+08]]

syn = [[5.10216E-04,   0.005102158,    0.05102158,    0.5102158],
[1.5489006,    15.48900604,    154.8900604,    1548.900604],
[1.45,  1.45E+01,  1.45E+02,  1.45E+03],
[1.08E+02,  1.08E+03,  1.08E+04,  1.08E+5],
[711.1086968,    7111.086968,    71110.86968,    711108.6968]]

colors = ['#5b9bd5', '#7f7f7f', '#ff0000', '#ffc000', '#70ad47' ]

methods = ['OURS', 'PALE-MLP', 'IONE', 'MNA', 'FRUI-P']

w2d = np.array(w2d)
d2w = np.array(d2w)
syn = np.array(syn)

dim = 4
w = 0.65
dimw = w / dim
gap = .1 / dim

fig, ax = plt.subplots()
x = np.arange(4)
for i in range(len(methods)):
    y = syn[i,:]
    b = ax.bar(x + i * (dimw+gap), y, dimw, bottom=0.001, label=methods[i], color=colors[i])

cnt = 1
for spine in plt.gca().spines.values():
    if cnt%2==0:
        spine.set_visible(False) #Indentation updated..
    cnt += 1

print x + (dimw+gap) / 2, map(str, x)
# ax.set_xticks([0.5, 1.5, 2.5, 3.5], map(str, x))
ax.set_xticks(x + 0.9 / 2)
ax.set_xticklabels([100, 1000, 10000, 100000])
ax.set_yscale('log')

ax.set_xlabel('Query number', fontsize=22)
ax.set_ylabel('Time cost (sec)', fontsize=22)

# ax.yaxis.grid(True, color='#d9d9d9')
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.125), fancybox=False, shadow=False, ncol=5)
leg.get_frame().set_linewidth(0.0)

plt.show()