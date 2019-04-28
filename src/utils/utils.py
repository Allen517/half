import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import json

def read_features(feat_file, delimiter=','):
    features = list()
    lookup = dict()
    look_back = list()
    with open(feat_file, 'r') as emb_handler:
        idx = 0
        for ln in emb_handler:
            ln = ln.strip()
            if ln:
                elems = ln.split(delimiter)
                if len(elems)==2:
                    continue
                # print('feature length:',len(elems))
                features.append(list(map(float, elems[1:])))
                lookup[elems[0]] = idx
                look_back.append(elems[0])
                idx += 1
    assert features, 'Fail to read features. The delimiter may be mal-used!'
    
    return np.array(features), lookup, look_back

def load_train_valid_labels(filename, lookups, valid_prop, delimiter=','):
    lbs = dict()
    lbs['src2end'] = dict()
    lbs['src2end']['train'] = defaultdict(list)
    lbs['src2end']['valid'] = defaultdict(list)
    lbs['end2src'] = dict()
    lbs['end2src']['train'] = defaultdict(list)
    lbs['end2src']['valid'] = defaultdict(list)
    with open(filename, 'r') as fin:
        for ln in fin:
            elems = ln.strip().split(delimiter)
            if len(elems)!=2:
                continue
            nd_src,nd_end = elems
            if nd_src not in lookups['src']:
                continue
            if nd_end not in lookups['end']:
                continue
            if np.random.random()<valid_prop:
                lbs['src2end']['train'][nd_src].append(nd_end)
                lbs['end2src']['train'][nd_end].append(nd_src)
            else:
                lbs['src2end']['valid'][nd_src].append(nd_end)
                lbs['end2src']['valid'][nd_end].append(nd_src)
    assert lbs['src2end']['train'] and lbs['end2src']['train'],\
            'Fail to read labels. The delimiter may be mal-used!'
    return lbs

def batch_iter(lbs, batch_size, neg_ratio, lookup, lb_tag_src, lb_tag_end):
    train_lbs = lbs['{}2{}'.format(lb_tag_src,lb_tag_end)]['train']
    train_lbs_inv = lbs['{}2{}'.format(lb_tag_end,lb_tag_src)]['train']
    # cands_end = list(lookup[lb_tag_end].keys())

    start_index = 0
    train_size = len(train_lbs)
    end_index = min(start_index+batch_size, train_size)

    lb_keys_src = list(train_lbs.keys())
    lb_keys_end = list(train_lbs_inv.keys())
    shuffle_indices = np.random.permutation(np.arange(train_size))
    while start_index < end_index:
        pos = {lb_tag_src:list(), lb_tag_end:list()}
        neg = {lb_tag_src:list(), lb_tag_end:list()}
        for i in range(start_index,end_index):
            idx = shuffle_indices[i]
            src_lb = lb_keys_src[idx]
            if not src_lb in lookup[lb_tag_src]:
                continue
            nd_idx = {lb_tag_src:-1, lb_tag_end:-1, 'rand':-1}
            nd_idx[lb_tag_src] = lookup[lb_tag_src][src_lb] # idx in src network
            lbs_idx_end = [lookup[lb_tag_end][lb_end] for lb_end in train_lbs[src_lb]]
            for lb_idx in lbs_idx_end:
                nd_idx[lb_tag_end] = lb_idx
                neg_idx_cur = {lb_tag_src:list(), lb_tag_end:list()}
                for k in range(neg_ratio):
                    nd_idx['rand'] = -1
                    while nd_idx['rand']<0 or nd_idx['rand'] in lbs_idx_end:
                        nd_idx['rand'] = lookup[lb_tag_end][lb_keys_end[np.random.randint(0, len(lb_keys_end))]]
                    neg_idx_cur[lb_tag_src].append(nd_idx[lb_tag_src])
                    neg_idx_cur[lb_tag_end].append(nd_idx['rand'])
                pos[lb_tag_src].append(nd_idx[lb_tag_src])
                pos[lb_tag_end].append(nd_idx[lb_tag_end])
                neg[lb_tag_src].append(neg_idx_cur[lb_tag_src])
                neg[lb_tag_end].append(neg_idx_cur[lb_tag_end])

        start_index = end_index
        end_index = min(start_index+batch_size, train_size)
        
        yield pos,neg

def valid_iter(lbs, valid_sample_size, lookup, lb_tag_src, lb_tag_end):
    valid_lbs = lbs['{}2{}'.format(lb_tag_src,lb_tag_end)]['valid']
    valid_lbs_inv = lbs['{}2{}'.format(lb_tag_end,lb_tag_src)]['train']
    cands_end = list(lookup[lb_tag_end].keys())

    valid = {lb_tag_src:list(), lb_tag_end:list()}
    lb_keys_src = list(valid_lbs.keys())
    lb_keys_end = list(valid_lbs_inv.keys())
    for lb_src in lb_keys_src:
        if not lb_src in lookup[lb_tag_src]:
            continue
        nd_idx = {lb_tag_src:-1, lb_tag_end:-1, 'rand':-1}
        nd_idx[lb_tag_src] = lookup[lb_tag_src][lb_src] # idx in src network
        lbs_idx_end = [lookup[lb_tag_end][lb_end] for lb_end in valid_lbs[lb_src]]
        for lb_idx in lbs_idx_end:
            nd_idx[lb_tag_end] = lb_idx
            cand = {lb_tag_src:list(),lb_tag_end:list()}
            cand[lb_tag_src].append(nd_idx[lb_tag_src])
            cand[lb_tag_end].append(nd_idx[lb_tag_end])
            for k in range(valid_sample_size-1):
                nd_idx['rand'] = -1
                while nd_idx['rand']<0 or nd_idx['rand'] in lbs_idx_end:
                    # nd_idx['rand'] = lookup[lb_tag_end][lb_keys_end[np.random.randint(0, len(lb_keys_end))]]
                    nd_idx['rand'] = np.random.randint(0, len(cands_end))
                cand[lb_tag_src].append(nd_idx[lb_tag_src])
                cand[lb_tag_end].append(nd_idx['rand'])
            if (cand[lb_tag_src] and cand[lb_tag_end])\
                and len(cand[lb_tag_src])==valid_sample_size\
                and len(cand[lb_tag_end])==valid_sample_size:
                valid[lb_tag_src].append(cand[lb_tag_src])
                valid[lb_tag_end].append(cand[lb_tag_end])
    # print('valid',valid)

    return valid

def write_in_file(filename, vec, tag):
    with open(filename, 'a+') as res_handler:
        if len(vec.shape)>1:
            column_size = vec.shape[1]
        else:
            column_size = 1
        reshape_vec = vec.reshape(-1)
        vec_size = len(reshape_vec)
        res_handler.write(tag+'\n')
        for i in range(0,vec_size,column_size):
            res_handler.write('{}\n'.format(' '.join([str(reshape_vec[i+k]) for k in range(column_size)])))

def save_dict(filepath, d_obj):
    with open(filepath, 'w') as fin:
        json.dump(d_obj, fin)

def save_index(filepath, indices):

    def save_dim(filepath_dim, dict_list):
        with open(filepath_dim, 'w') as fout:
            for dim_dict in dict_list:
                tmp_dict = dict()
                for key,value in dim_dict.items():
                    tmp_dict[str(key)] = list(value)
                fout.write(json.dumps(tmp_dict)+'\n')

    for nm in range(len(indices)):
        save_dim('%s.m%d.src'%(filepath,nm), indices[nm]['src'])
        save_dim('%s.m%d.end'%(filepath,nm), indices[nm]['end'])

def save_res(filepath, model_res):

    for nm in range(len(model_res)):
        for tag in ['src', 'end']:
            tmp_dict = dict()
            for k,v in model_res[nm][tag].items():
                tmp_dict[str(k)] = list(map(lambda x: int(x), v))
            with open('%s.m%d.%s'%(filepath,nm,tag), 'w+') as fout:
                fout.write(json.dumps(tmp_dict))