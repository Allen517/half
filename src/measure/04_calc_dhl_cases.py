import sys,os
import random
import time

def read_code(filename):
	print 'read {}'.format(filename)
	code_map = dict()
	with open(filename, 'r') as fin:
		for ln in fin:
			elems = ln.strip().split('\t')
			if len(elems)<2:
				continue
			code_map[elems[0]]=elems[1]
	return code_map

# def match_num(src_code, target_code):
# 	xor_res = int(src_code, 2)^int(target_code, 2)
# 	one_cnt = 0
# 	while xor_res != 0:
# 		s = xor_res%2
# 		if s==1:
# 			one_cnt += 1
# 		xor_res /= 2
# 	return one_cnt

def find_matched(code, target_code_map, thres):
	match_res = set()
	for target_key, target_code in target_code_map.iteritems():
		one_cnt = hamming(code, target_code)
		if one_cnt<=thres:
			match_res.add(target_key)
	return match_res

def hamming(src_code, target_code):
	assert len(src_code) == len(target_code)
	return sum(c1 != c2 for c1, c2 in zip(src_code, target_code))

def read_test_anchors(filename):
	anchors = list()
	with open(filename, 'r') as fin:
		for ln in fin:
			elems = ln.strip().split()
			for val in elems[1].split(';'):
				anchors.append((elems[0],val))
	return anchors

if __name__=='__main__':
	outfile = 'match.res'
	if os.path.exists(outfile):
		os.remove(outfile)
	douban_code_map = read_code('dhl-alp.case.model.code.f')
	weibo_code_map = read_code('dhl-alp.case.model.code.g')
	anchors = read_test_anchors('data/douban2weibo/d2w.anchor_links.labels.0.9.test')
	t1 = time.time()
	with open(outfile, 'w') as fout:
		for d_key, w_key in anchors:
			true_cnt = hamming(douban_code_map[d_key], weibo_code_map[w_key])
			match_res = find_matched(douban_code_map[d_key], weibo_code_map, true_cnt)
			if true_cnt==0:
				print '{},{}:{},{}'.format(d_key, w_key, true_cnt, len(match_res))
			fout.write('{},{}:{},{}\n'.format(d_key, w_key, true_cnt, len(match_res)))
			fout.flush()
	t2 = time.time()
	print t2-t1

