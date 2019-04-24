# -*- coding:utf8 -*-

import pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

v=3
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
# fig_height = fig_width*golden_mean      # height in inches
fig_height = fig_width*0.8
fig_size =  [fig_width*v,fig_height*v]
params = {'backend': 'ps',
          'axes.labelsize': 30,
          'text.fontsize': 30,
          'legend.fontsize': 30,
          'xtick.labelsize': 30,
          'ytick.labelsize': 30,
          'lines.linewidth': 4,
          'text.usetex': True,
          'font.family':'Times New Roman',
          'figure.figsize': fig_size}
pylab.rcParams.update(params)

# base_comp_time = 4.e-06
# net_size = 1e6

# k = np.arange(1,1e2, .001)*1e8

# t_pair = k*net_size*base_comp_time/60.

# t_tree = k*np.log(net_size)*base_comp_time/60.

# t_hash = k*base_comp_time/60.

base_comp_time = 4.e-05/60.

k = np.arange(1,2.1e2, .001)*1e7

t_pair = k**2*base_comp_time

t_tree = k*np.log(k)*base_comp_time

t_hash = k*base_comp_time

fig, ax = plt.subplots()

# print np.where(t_tree>100)[0]
# print np.where(t_pair>60*24)[0]
# print t_pair[100]
# print t_pair[0]
# print t_tree[100]
# print t_pair[100]

ax.semilogy(k, t_hash, '-', label='hash-based search', color='blue')
ax.semilogy(k, t_tree, '--', label='tree-based search', color='orange')
ax.semilogy(k, t_pair, ':', label='pairwise search', color='green')

x=[140e6, 350e6, 1.5e9]

for val in x:
	ax.axvline(x=val, linestyle='--', color='grey', linewidth=2) 

	ax.axhline(y=val**2*base_comp_time, xmax=val/2.1e9, linestyle='--', color='green', linewidth=2)
	ax.axhline(y=val*np.log(val)*base_comp_time, xmax=val/2.1e9, linestyle='--', color='orange', linewidth=2)
	ax.axhline(y=val*base_comp_time, xmax=val/2.1e9, linestyle='--', color='blue', linewidth=2)
	print val
	print val**2*base_comp_time, val*np.log(val)*base_comp_time, val*base_comp_time

ax.get_yaxis().set_visible(False)

# ax.semilogy(k, t_hash, '-', label='$O(n)$')
# ax.semilogy(k, t_tree, '--', label='$O(n \log n)$')
# ax.semilogy(k, t_pair, ':', label='$O(n^2)$')

# plt.xlabel(r'\#Matching Tasks ($\times 10^6$)')
plt.xlabel(r'Scale of social network')
# plt.ylabel('Time (Minutes)')

# plt.yticks(np.arange(1, 1e5, 100))
# plt.ylim(1, 1e2)
plt.xlim(1e7, 2.1e9)


# scale_x = 1e8
# ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
# ax.xaxis.set_major_formatter(ticks_x)

ax.legend(loc=5)

plt.show()