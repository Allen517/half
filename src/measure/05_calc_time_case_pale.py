import sys,os
import random
import time
import numpy as np

def read_code(filename):
	code_map = dict()
	with open(filename, 'r') as fin:
		for ln in fin:
			elems = ln.strip().split('\t')
			if len(elems)<2:
				continue
			code_map[elems[0]]=np.array(map(float, elems[1].split(',')))
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

def find_matched(code, target_code_map):
	match_res = set()
	min_dist = 1000
	min_key = ''
	for target_key, target_code in target_code_map.iteritems():
		dist = geo_distance(code, target_code)
		if dist<=min_dist:
			min_dist = dist
			min_key = target_key
	return min_dist, target_key

def geo_distance(vec1, vec2):
    return .5*np.sum((vec1-vec2)**2)

def hamming(src_code, target_code):
	assert len(src_code) == len(target_code)
	return sum(c1 != c2 for c1, c2 in zip(src_code, target_code))

if __name__=='__main__':
	outfile = 'match.res'
	if os.path.exists(outfile):
		os.remove(outfile)
	douban_code_map = read_code('dhl-alp.case.model.code.pale.f')
	weibo_code_map = read_code('dhl-alp.case.model.code.pale.g')
	num = 10
	t1 = time.time()
	with open(outfile, 'w') as fout:
		douban_keys = douban_code_map.keys()
		for i in range(num):
			d_key = douban_keys[random.randint(0,len(douban_keys)-1)]
			target_dist, target_key = find_matched(douban_code_map[d_key], weibo_code_map)
			print '{},{}:{}'.format(d_key, target_key,target_dist)
			fout.write('{},{}:{}\n'.format(d_key, target_key,target_dist))
	t2 = time.time()
	print t2-t1