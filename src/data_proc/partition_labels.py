
import numpy as np
import sys

def partition_labels(filepath, range_from, range_to, gap):

	for p in np.arange(range_from, range_to, gap):
		outFilePath = '{}.labels.{}'.format(filepath, p)
		with open(filepath, 'r') as in_handler, open(outFilePath+'.train', 'w') as out_handler\
				, open(outFilePath+'.test', 'w') as out_handler_test:
			for ln in in_handler:
				wrtLn = ''
				wrtLn_test = ''
				nds = ln.strip().split(' ')
				if len(nds)<2:
					continue
				d_nd = nds[0]
				w_nd = nds[1]
				if np.random.rand()<p:
					wrtLn += d_nd+' '+w_nd+'\n'
				else:
					wrtLn_test += d_nd+' '+w_nd+'\n'
				out_handler.write(wrtLn)
				out_handler_test.write(wrtLn_test)

if __name__=='__main__':
	if len(sys.argv)<5:
		print 'please input objective label file, range_from, range_to, gap'
		sys.exit(1)
	partition_labels(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))