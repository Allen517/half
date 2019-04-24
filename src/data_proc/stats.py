nds = set()

with open('../../data/weibo_links.txt', 'r') as objF:
	cnt = 0
	for ln in objF:
		elems = ln.strip().split()
		cnt += len(elems)-1
		nds.update(elems)

print cnt, len(nds)