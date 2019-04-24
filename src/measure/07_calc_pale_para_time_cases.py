import sys,os
import random
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures

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

def main_proc(num):
	print num
	for i in range(num):
		douban_keys = douban_code_map.keys()
		d_key = douban_keys[random.randint(0,len(douban_keys)-1)]
		target_key, target_dist = find_matched(douban_code_map[d_key], weibo_code_map)
		print '{},{}:{}'.format(d_key, target_key,target_dist)

if __name__=='__main__':
	douban_code_map = read_code('dhl-alp.case.model.code.pale.f')
	weibo_code_map = read_code('dhl-alp.case.model.code.pale.g')
	overall = 100
	worker = 8
	t1 = time.time()
	with ThreadPoolExecutor(max_workers=worker) as executor:
		future_to_proc = {executor.submit(main_proc, num)
				:overall/worker if num+overall/worker<overall else overall-num for num in range(int(overall/worker))}
		for future in futures.as_completed(future_to_proc):
			print 'Finish processing %d'%(future_to_proc[future])
			if future.exception() is not None:
				print future.exception()
	t2 = time.time()
	print t2-t1