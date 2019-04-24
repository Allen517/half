from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import sys,os
import random
import time

douban_code_map = dict()
weibo_code_map = dict()

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

def main_proc(idx, batch_size, fout):
	proc_num = 0
	print 'start to matching in proc {}'.format(idx)
	weibo_keys = weibo_code_map.keys()
	max_idx = len(weibo_keys) if (idx+1)*batch_size>len(weibo_keys) else (idx+1)*batch_size
	for w_key in weibo_keys[idx*batch_size:max_idx]:
		match_res = find_matched(weibo_code_map[w_key], douban_code_map, int(sys.argv[1]))
		if match_res:
			wrtLn = ''
			print len(match_res)
			for res in match_res:
				wrtLn += res+','
			wrtLn = wrtLn[:-1]+'\n'
			proc_num += 1
			fout.write(wrtLn)
			print 'Finish matching {} users in proc {}'.format(proc_num, idx)
	fout.write(wrtLn)

if __name__=='__main__':
	if len(sys.argv)<2:
		print 'please input threshold, outfile'
		sys.exit(1)
	outfile = sys.argv[2]
	fout = open(outfile, 'w')
	if os.path.exists(outfile):
		os.remove(outfile)
	douban_code_map = read_code('dhl-alp.sys.model.code.f')
	weibo_code_map = read_code('dhl-alp.sys.model.code.g')
	overall = len(weibo_code_map)
	worker = 8
	t1 = time.time()
	with ThreadPoolExecutor(max_workers=worker) as executor:
		future_to_proc = {executor.submit(main_proc, idx, overall/worker, fout)
				:idx for idx in range(int(overall/worker))}
		for future in futures.as_completed(future_to_proc):
			print 'Finish processing %d'%(future_to_proc[future])
			if future.exception() is not None:
				print future.exception()
	t2 = time.time()
	print 'time cost:',t2-t1
