import sys,os

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

if __name__=='__main__':
	if len(sys.argv)<3:
		print 'please input query node'
		sys.exit(1)
	douban_code_map = read_code('dhl-alp.sys.model.code.f')
	weibo_code_map = read_code('dhl-alp.sys.model.code.g')
	if sys.argv[1] in douban_code_map:
		match_res = find_matched(douban_code_map[sys.argv[1]], weibo_code_map, int(sys.argv[2]))
		print len(match_res)
		if match_res:
			with open('match.res', 'w') as fout:
				fout.write(sys.argv[1]+'\n')
				for res in match_res:
					fout.write(res+',')
