from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import sys,os
import random
import time

def read_code(filename):
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

def main_proc(num):
	print num 
	for i in range(num):
		douban_keys = douban_code_map.keys()
		d_key = douban_keys[random.randint(0,len(douban_keys)-1)]
		match_res = find_matched(douban_code_map[d_key], weibo_code_map, int(sys.argv[1]))
		print d_key,len(match_res)
		if len(match_res)>50:
			continue
		if match_res:
			wrtLn = ''
			for res in match_res:
				wrtLn += res+','
			print wrtLn[:-1]

if __name__=='__main__':
	if len(sys.argv)<2:
		print 'please input threshold'
		sys.exit(1)
	outfile = 'match.res'
	if os.path.exists(outfile):
		os.remove(outfile)
	douban_code_map = read_code('dhl-alp.case.model.code.f')
	weibo_code_map = read_code('dhl-alp.case.model.code.g')
	overall = 100
	worker = 8
	t1 = time.time()
	with ThreadPoolExecutor(max_workers=worker) as executor:
		future_to_proc = {executor.submit(main_proc, overall/worker)
				:num for num in range(int(overall/worker))}
		for future in futures.as_completed(future_to_proc):
			print 'Finish processing %d'%(future_to_proc[future])
			if future.exception() is not None:
				print future.exception()
	t2 = time.time()
	print t2-t1
