# -*- coding=UTF-8 -*-\n
import numpy as np
import random
from collections import defaultdict
import json
import re,os

def get_model_perf(out_file):
    # eval_hashing_model('d2w_embedding/douban_all.txt', 'd2w_embedding/weibo_all.txt', 'case_model/d2w.alp_model.case.dcnh-sp.times_5.epoch_4300', out_file, calc_mlp_sp_res)
    # eval_pale_model('d2w_embedding/16/douban_all.txt', 'd2w_embedding/16/weibo_all.txt', 'anchor_res/w2d.sys.alp_model.dcnh-sp.train_0.9.times_4.epoch_2100', out_file+'.pale', calc_mlp_sp_res)
    eval_hashing_model('d2w_embedding/16/douban_all.txt', 'd2w_embedding/16/weibo_all.txt', 'anchor_res/w2d.sys.alp_model.dcnh-sp.train_0.9.times_4.epoch_2100', out_file, calc_mlp_sp_res)

def _read_embeddings(filename):
    if not os.path.exists(filename):
        return None
    print 'reading {}'.format(filename)
    embedding = dict()
    with open(filename, 'r') as f_handler:
        for ln in f_handler:
            ln = ln.strip()
            if ln:
                elems = ln.split()
                if len(elems)==2:
                    continue
                embedding[elems[0]] = map(float, elems[1:])
    return embedding

def _read_model(filename):
    print 'reading model {}'.format(filename)
    if not os.path.exists(filename):
        return None
    model = defaultdict(list)
    with open(filename, 'r') as f_handler:
        cur_key = ''
        for ln in f_handler:
            ln = ln.strip()
            if 'h' in ln or 'b' in ln or 'out' in ln:
                cur_key = ln
                continue
            model[cur_key].append(map(float, ln.split()))
    return model

def _tanh(mat):
    # (1-exp(-2x))/(1+exp(-2x))
    mat = np.array(mat)
    return np.tanh(mat)

def _sigmoid(mat):
    # 1/(1+exp(-mat))
    mat = np.array(mat)
    return 1./(1+np.exp(-mat))

def _hamming_distance(vec1, vec2):
    res = np.where(vec1*vec2<0, np.ones(vec1.shape), np.zeros(vec1.shape))
    return np.sum(res)

def _dot_distance(vec1, vec2):
    return -np.sum(vec1*vec2)

def _geo_distance(vec1, vec2):
    return .5*np.sum((vec1-vec2)**2)

def calc_mlp_sp_res(inputs, n_layer, model, model_type):
    inputs = np.array(inputs)

    layer = _sigmoid(np.dot(inputs,np.array(model['h0_'+model_type]))+np.array(model['b0_'+model_type]).reshape(-1))
    for i in range(1, n_layer):
        layer = _sigmoid(np.dot(layer,np.array(model['h{}'.format(i)]))+np.array(model['b{}'.format(i)]).reshape(-1))
    out = _tanh(np.dot(layer,np.array(model['out']))+np.array(model['b_out']).reshape(-1))

    return out

def calc_mlp_res(inputs, n_layer, model):
    inputs = np.array(inputs)

    layer = _sigmoid(np.dot(inputs,np.array(model['h0']))+np.array(model['b0']).reshape(-1))
    for i in range(1, n_layer):
        layer = _sigmoid(np.dot(layer,np.array(model['h{}'.format(i)]))+np.array(model['b{}'.format(i)]).reshape(-1))
    out = _tanh(np.dot(layer,np.array(model['out']))+np.array(model['b_out']).reshape(-1))

    return out

def write_map_code(embedding, model, calc_type, out_file, calc_model_res):
    with open(out_file+'.'+calc_type, 'w') as fout:
        fout.write(calc_type+"\n")
        cnt = 0
        wrtLn = ''
        for key in embedding.keys():
            model_res = calc_model_res(embedding[key], 1, model, calc_type)
            wrtLn += key+'\t'+''.join('1' if i>0 else '0' for i in model_res.reshape(-1))+'\n'
            if cnt%1000==0:
                fout.write(wrtLn)
                wrtLn = ''

def eval_hashing_model(src_emb_file, target_emb_file, model_file, out_file, calc_model_res):
    print 'processing {} and {}'.format(src_emb_file, target_emb_file)
    if not os.path.exists(src_emb_file) or not os.path.exists(target_emb_file):
        print 'file not found...'
        return

    src_embedding = _read_embeddings(src_emb_file)
    target_embedding = _read_embeddings(target_emb_file)

    model = _read_model(model_file)
    if not model:
        return 

    write_map_code(src_embedding, model, 'f', out_file, calc_model_res)
    write_map_code(target_embedding, model, 'g', out_file, calc_model_res)

def write_map_continu_code(embedding, model, calc_type, out_file, calc_model_res):
    with open(out_file+'.'+calc_type, 'w') as fout:
        fout.write(calc_type+"\n")
        cnt = 0
        wrtLn = ''
        for key in embedding.keys():
            model_res = calc_model_res(embedding[key], 1, model, calc_type)
            wrtLn += key+'\t'+','.join(str(i) for i in model_res.reshape(-1))+'\n'
            if cnt%1000==0:
                fout.write(wrtLn)
                wrtLn = ''

def eval_pale_model(src_emb_file, target_emb_file, model_file, out_file, calc_model_res):
    print 'processing {} and {}'.format(src_emb_file, target_emb_file)
    if not os.path.exists(src_emb_file) or not os.path.exists(target_emb_file):
        print 'file not found...'
        return

    src_embedding = _read_embeddings(src_emb_file)
    target_embedding = _read_embeddings(target_emb_file)

    model = _read_model(model_file)
    if not model:
        return 

    write_map_continu_code(src_embedding, model, 'f', out_file, calc_model_res)
    write_map_continu_code(target_embedding, model, 'g', out_file, calc_model_res)

# get_model_perf('dhl-alp.case.model.code')
get_model_perf('dhl-alp.sys.model.code')

