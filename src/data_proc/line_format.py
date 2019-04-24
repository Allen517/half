# -*- coding:utf8 -*-

import numpy as np

def format_net(filepath, outfile):
	with open(filepath, 'r') as fin, open(outfile, 'w') as fout\
			, open(outfile+'.reverse', 'w') as fout_reverse:
		wrtLn = ''
		wrtLn_reverse = ''
		cnt = 0
		for ln in fin:
			elems = ln.strip().split()
			nd_src = elems[0]
			for nd_target in elems[1:]:
				wrtLn += nd_src+' '+nd_target+'\n'
				wrtLn_reverse += nd_target+' '+nd_src+'\n'
				cnt += 1
				if cnt%10000==0:
					fout.write(wrtLn)
					fout_reverse.write(wrtLn_reverse)
					wrtLn = ''
					wrtLn_reverse = ''
		fout.write(wrtLn)
		fout_reverse.write(wrtLn_reverse)

net_dir = 'toy_dat/subnets/'
format_dir = 'toy_dat/subnets/graph_in_line/'

for s in np.arange(.2, 1., .1):
	format_net('{}{}{}.{}'.format(net_dir,'blogcatalog_net.src.s_',s,'c_0.8')\
				, '{}{}{}.{}'.format(format_dir,'blogcatalog_net.line_format.src.s_',s,'c_0.8'))
	format_net('{}{}{}.{}'.format(net_dir,'blogcatalog_net.target.s_',s,'c_0.8')\
				, '{}{}{}.{}'.format(format_dir,'blogcatalog_net.line_format.target.s_',s,'c_0.8'))

for c in np.arange(.1, 1., .1):
	format_net('{}{}{}'.format(net_dir,'blogcatalog_net.src.s_0.5.c_',c)\
				, '{}{}{}'.format(format_dir,'blogcatalog_net.line_format.src.s_0.5.c_',c))
	format_net('{}{}{}'.format(net_dir,'blogcatalog_net.target.s_0.5.c_',c)\
				, '{}{}{}'.format(format_dir,'blogcatalog_net.line_format.target.s_0.5.c_',c))