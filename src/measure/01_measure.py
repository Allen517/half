import numpy as np
import re

filename = 'mrr.train_var.09eval.w2d'
outfile = 'mrr.train_var.09eval.w2d.stats'
with open(filename, 'r') as fin, open(outfile, 'w') as fout:
	wrtLn = ''
	for ln in fin:
		ln = ln.strip()
		reg1 = r'alp_model\.(.*?)\.train_(.*?)\.times_(.*?)\.epoch'
		reg2 = r'mean_mrr:(.*?), var:(.*)'

		pattern1 = re.compile(reg1)
		pattern2 = re.compile(reg2)

		match1 = pattern1.search(ln)
		match2 = pattern2.search(ln)

		if match1:
			model = match1.group(1)
			train_prop = match1.group(2)
			times = match1.group(3)
			# print match1.groups()
			wrtLn += model+'.'+'times_'+times+'\t'+train_prop+'\t'
		if match2:
			wrtLn += match2.group(1)+'\t'+match2.group(2)
			fout.write(wrtLn+'\n')
			wrtLn = ''


