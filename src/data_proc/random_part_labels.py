# -*- coding:utf8 -*-

import sys
import random
import numpy as np

def random_partition_labels(filepath, range_from, range_to, gap):

	for p in np.arange(range_from, range_to, gap):
		outFilePath = '{}.labels.{}'.format(filepath, p)
		with open(filepath, 'r') as in_handler, open(outFilePath+'.train', 'w') as out_handler\
				, open(outFilePath+'.test', 'w') as out_handler_test:
			from_nds = set()
			to_nds = set()
			for ln in in_handler:
				nds = ln.strip().split(' ')
				if len(nds)<2:
					continue
				d_nd = nds[0]
				w_nd = nds[1]
				from_nds.add(d_nd)
				to_nds.add(w_nd)
			from_nds = list(from_nds)
			to_nds = list(to_nds)
			for f_nd in from_nds:
				wrtLn = ''
				wrtLn_test = ''
				if random.random()<p:
					wrtLn += f_nd+' '+to_nds[random.randint(0,len(to_nds)-1)]+'\n'
				else:
					wrtLn_test += f_nd+' '+to_nds[random.randint(0,len(to_nds)-1)]+'\n'

				out_handler.write(wrtLn)
				out_handler_test.write(wrtLn_test)

if __name__=='__main__':
	if len(sys.argv)<5:
		print 'please input objective label file, range_from, range_to, gap'
		sys.exit(1)
	random_partition_labels(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))