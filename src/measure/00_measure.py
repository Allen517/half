import numpy as np
import re

filename = 'mrr.06eval.d2w'
outfile = 'mrr.06eval.d2w.stats'
with open(filename, 'r') as fin, open(outfile, 'w') as fout:
	wrtLn = ''
	for ln in fin:
		ln = ln.strip()
		reg1 = r'alp_model\.(.*?)\.train_(.*?)\.s_(.*?)\.c_(.*?).epoch|alp_model\.(.*?)\.s_(.*?)\.c_(.*?)\.epoch'
		reg2 = r'mean_mrr:(.*?), var:(.*)'

		pattern1 = re.compile(reg1)
		pattern2 = re.compile(reg2)

		match1 = pattern1.search(ln)
		match2 = pattern2.search(ln)

		if match1:
			if match1.group(2)!=None:
				model = match1.group(1)
				train_prop = match1.group(2)
				s = match1.group(3)
				c = match1.group(4)
				wrtLn += model+'\t'+train_prop+'\t'+s+'\t'+c+'\t'
			else:
				model = match1.group(5)
				s = match1.group(6)
				c = match1.group(7)
				wrtLn += model+'\t0.6\t'+s+'\t'+c+'\t'
		if match2:
			wrtLn += match2.group(1)+'\t'+match2.group(2)
			fout.write(wrtLn+'\n')
			wrtLn = ''


