# -*- coding=UTF-8 -*-\n
from __future__ import print_function

import numpy as np
import random
from collections import defaultdict
import json
import re,os

from utils.utils import *
from eval.measures import *

class Eval(object):

    def __init__(self):
        self.model = defaultdict(dict)
        self.labels = list()
        self.inputs = defaultdict(dict)

    def _read_model(self, **kwargs):
        raise NotImplementedError

    def _calc_model_res(self, **kwargs):
        raise NotImplementedError

    def _read_inputs(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Inputs file does not exist: %s'%filepath

        inputs = dict()
        with open(filepath, 'r') as f_handler:
            for ln in f_handler:
                ln = ln.strip()
                if ln:
                    elems = ln.split(',')
                    if len(elems)==2:
                        continue
                    inputs[elems[0]] = list(map(float, elems[1:]))
        return inputs

    def _read_labels(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Label file does not exist: %s'%filepath

        lbs = {
            'src2end': defaultdict(list),
            'end2src': defaultdict(list)
        }
        with open(filepath, 'r') as fin:
            for ln in fin:
                elems = ln.strip().split(',')
                if len(elems)!=2:
                    continue
                nd_src,nd_end = elems
                lbs['src2end'][nd_src].append(nd_end)
                lbs['end2src'][nd_end].append(nd_src)
        return lbs

    def _init_eval(self, **kwargs):
        allows_keys = {'feat_src', 'feat_end', 'linkage'}
        for k in kwargs.keys():
            assert k in allows_keys, 'Invalid file inputs: '+k

        print('processing {} and {}'.format(kwargs['feat_src'], kwargs['feat_end']))
        assert os.path.exists(kwargs['feat_src']) and os.path.exists(kwargs['feat_end'])\
                , 'Files not found: %s, %s'%(kwargs['feat_src'], kwargs['feat_end'])

        self.inputs['src'] = self._read_inputs(kwargs['feat_src'])
        self.inputs['end'] = self._read_inputs(kwargs['feat_end'])
        assert self.inputs['src'], 'Failed to read features from source network'
        assert self.inputs['end'], 'Failed to read features from end network'

        self.labels = self._read_labels(kwargs['linkage'])
        assert self.labels, 'Failed to read labels'

    def calc_mrr_by_dist(self, **kwargs):
        pass

    def choose_candidates(self, **kwargs):
        pass

class Eval_HALF_DP(Eval):

    def __init__(self, model_type):
        super(Eval_HALF_DP, self).__init__()

        assert model_type in {'lin', 'mlp'}, 'Model type must be lin/mlp'
        self.model_type = model_type
        self.models = list()

    def _read_models(self, **kwargs):
        allows_keys = {'filepath', 'n_model'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        models = list()
        for i in range(1,kwargs['n_model']+1):
            model = self._read_model(filepath='%s.times%d'%(kwargs['filepath'],i))
            assert model, 'Failed to read model'

            models.append(model)

        return models

    def _read_model(self, **kwargs):
        allows_keys = {'filepath'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        print('reading model %s'%(kwargs['filepath']))
        assert os.path.exists(kwargs['filepath']), 'Model file does not exist: %s'%kwargs['filepath']

        model = defaultdict(list)
        with open(kwargs['filepath'], 'r') as f_handler:
            cur_key = ''
            for ln in f_handler:
                ln = ln.strip()
                if 'h' in ln or 'b' in ln or 'out' in ln:
                    cur_key = ln
                    continue
                model[cur_key].append(list(map(float, ln.split())))
        return model

    def _calc_model_lin_res(self, **kwargs):
        allows_keys = {'tag', 'inputs', 'n_model'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        calc_tag = kwargs['tag']
        inputs = np.array(kwargs['inputs'])
        n_model = kwargs['n_model']

        out = tanh(
                np.dot(inputs,np.array(self.models[n_model]['out_'+calc_tag]))
                    +np.array(self.models[n_model]['b_out_'+calc_tag]).reshape(-1)
                )

        return out

    def _calc_model_mlp_res(self, **kwargs):
        allows_keys = {'tag', 'inputs', 'n_layer', 'n_model'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        calc_tag = kwargs['tag']
        inputs = np.array(kwargs['inputs'])
        n_layer = kwargs['n_layer']
        n_model = kwargs['n_model']

        layer = sigmoid(np.dot(inputs,np.array(self.models[n_model]['h0_'+calc_tag]))+np.array(self.models[n_model]['b0_'+calc_tag]).reshape(-1))
        for i in range(1, n_layer):
            layer = sigmoid(np.dot(layer,np.array(self.models[n_model]['h{}_{}'.format(i,calc_tag)]))
                                +np.array(self.models[n_model]['b{}_{}'.format(i,calc_tag)]).reshape(-1))
        out = tanh(np.dot(layer,np.array(self.models[n_model]['out_'+calc_tag]))+np.array(self.models[n_model]['b_out_'+calc_tag]).reshape(-1))

        return out

    def _calc_model_res(self, **kwargs):
        if self.model_type=='lin':
            return self._calc_model_lin_res(**kwargs)

        return self._calc_model_mlp_res(**kwargs)

    def calc_mrr_by_dist(self, **kwargs):
        allows_keys = {'model', 'n_model', 'candidate_num', 'dist_calc', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw
        self.models = self._read_models(filepath=kwargs['model'], n_model=kwargs['n_model'])

        mrr_list = {
            'src2end': tuple(),
            'end2src': tuple()
        }

        with open(kwargs['out_file'], 'w') as fout:
            tps = ['src', 'end']
            tps_len = len(tps)-1
            for tp_id in range(len(tps)):
                cnt = 0
                wrt_lns = ''
                lb_tp = '%s2%s'%(tps[tp_id], tps[tps_len-tp_id])
                fout.write('%s\n'%lb_tp)
                to_keys = list(self.inputs[tps[tps_len-tp_id]].keys())
                to_size = len(to_keys)
                fout.write('Overall: %d\n'%len(self.labels[lb_tp].keys()))
                for nd_from, nds_to in self.labels[lb_tp].items():
                    for nd_to in nds_to:
                        anchor_dists = np.zeros(kwargs['n_model'])
                        noise_dists = np.zeros((kwargs['n_model'],kwargs['candidate_num']))
                        for nm in range(kwargs['n_model']):
                            model_res = {
                                'from': self._calc_model_res(inputs=self.inputs[tps[tp_id]][nd_from]
                                                                , tag=tps[tp_id], n_model=nm),
                                'to': self._calc_model_res(inputs=self.inputs[tps[tps_len-tp_id]][nd_to]
                                                                , tag=tps[tps_len-tp_id], n_model=nm),
                                'rand': None
                            }

                            anchor_dist = kwargs['dist_calc'](model_res['from'], model_res['to'])
                            anchor_dists[nm] = anchor_dist

                            rand_nds = set()
                            for k in range(kwargs['candidate_num']):
                                rand_nd_to = to_keys[np.random.randint(0, to_size)]
                                while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                                    rand_nd_to = to_keys[np.random.randint(0, to_size)]
                                rand_nds.add(rand_nd_to)
                                model_res['rand'] = self._calc_model_res(
                                                            inputs=self.inputs[tps[tps_len-tp_id]][rand_nd_to]
                                                            , tag=tps[tps_len-tp_id], n_model=nm
                                                            )
                                noise_dist = kwargs['dist_calc'](model_res['from'], model_res['rand'])
                                noise_dists[nm,k] = noise_dist

                        pred_pos = 1
                        mean_anchor_dist = np.mean(anchor_dists)
                        for k in range(kwargs['candidate_num']):
                            mean_noise_dist = np.mean(noise_dists[:,k])
                            if mean_anchor_dist>=mean_noise_dist:
                                pred_pos += 1
                        cur_mrr = 1./pred_pos
                        mrr_list[lb_tp] += cur_mrr,
                        cnt += 1
                        wrt_lns += '(%s,%s):%f;%f\n'%(nd_from, nd_to, cur_mrr, mean_anchor_dist)
                        if not cnt%10:
                            fout.write(wrt_lns)
                            print('Processing %d records'%cnt)
                            wrt_lns = ''
                if cnt%10:
                    fout.write(wrt_lns)
                fout.write('mean_mrr:{}, var:{}\n'
                        .format(np.mean(mrr_list[lb_tp]), np.var(mrr_list[lb_tp])))

    def build_index(self, **kwargs):
        allows_keys = {'model', 'n_model', 'n_dim'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid index builder parameter: '+kw

        self.models = self._read_models(filepath=kwargs['model'], n_model=kwargs['n_model'])

        n_dim = kwargs['n_dim']
        n_model = kwargs['n_model']

        dim_index = [{
            'src': [defaultdict(set) for i in range(n_dim)],
            'end': [defaultdict(set) for i in range(n_dim)]
        } for nm in range(kwargs['n_model'])]
        model_res = [{
            'src': defaultdict(list),
            'end': defaultdict(list)
        } for nm in range(kwargs['n_model'])]
        # model_res = {
        #     'src': [defaultdict(list) for i in range(n_model)],
        #     'end': [defaultdict(list) for i in range(n_model)]
        # }

        # Build indices
        print('Building indices...')
        for nm in range(n_model):
            print('Processing %d model'%nm)
            for tag in ['src', 'end']:
                cnt = 0
                for k,v in self.inputs[tag].items():
                    res = self._calc_model_res(inputs=v, tag=tag, n_model=nm)
                    res_bin = np.where(res>=0, np.ones(res.shape,dtype=int), -np.ones(res.shape,dtype=int))
                    model_res[nm][tag][k] = res_bin
                    for idx in range(len(res_bin)):
                        # print(nm, tag, idx, res_bin[idx], len(dim_index[nm][tag]))
                        dim_index[nm][tag][idx][res_bin[idx]].add(k)
                    cnt += 1
        print('Finish building indices')

        return dim_index, model_res

    def _search_sim_res(self, model_res, dim_index, search_nd, thres):
        search_res = defaultdict(int)

        for nm in range(len(dim_index)):
            cnt_search_res = defaultdict(int)
            res_nd_from = model_res[nm]['src'][search_nd]
            for dim_idx in range(len(res_nd_from)):
                if res_nd_from[dim_idx] in dim_index[nm]['end'][dim_idx]:
                    for nd in dim_index[nm]['end'][dim_idx][res_nd_from[dim_idx]]:
                        cnt_search_res[nd] += 1
            for k,v in cnt_search_res.items():
                if v>=thres:
                    search_res[k] += v
    #         print(search_res['585135'], cnt_search_res['585135'])
    #         print(search_res['2666863'], cnt_search_res['2666863'])
                            
        # print(search_res)
        search_res_list = list()
        for k,v in search_res.items():
            search_res_list.append([k,v])
            
        return search_res_list

    def choose_candidates(self, **kwargs):
        allows_keys = {'dim_index', 'model_res', 'filter_thres', 'candidate_num', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid candidates calculation parameter: '+kw

        dim_index = kwargs['dim_index']
        model_res = kwargs['model_res']
        filter_thres = kwargs['filter_thres']
        # col_prop = kwargs['col_prop']
        # n_model = kwargs['n_model']
        # n_dim = kwargs['n_dim']
        
        # dim_index, model_res = self._build_index(model=kwargs['model'], n_dim=n_dim, n_model=n_model)
        # save_index(kwargs['out_file']+'.index', dim_index)
        save_res(kwargs['out_file'], model_res)

        with open(kwargs['out_file'], 'w') as fout:
            # Searching
            cnt = 0
            hits = 0
            # cand_lens = []
            # hit_cand_lens = list()
            wrt_lns = ''
            for nd_from, nds_to in self.labels['src2end'].items():
                search_res = self._search_sim_res(model_res, dim_index, nd_from, filter_thres)

                sort_index = np.argsort(list(map(lambda x: x[1], search_res)))[::-1]
                filter_nds = list()
                filter_nds_set = set()
                filter_vals = list()
                hit_cnt = 0
                for idx in sort_index:
                    hit_cnt += 1
                    filter_nds.append(search_res[idx][0])
                    filter_vals.append(search_res[idx][1])
                    if hit_cnt>=kwargs['candidate_num']:
                        break
                filter_nds_set = set(filter_nds)

                cnt += 1
                is_hit = False
                for nd_to in nds_to:
                    if nd_to in filter_nds_set:
                        hits += 1
                        is_hit = True
                wrt_lns += '({},{}):{}:{}\n'.format(nd_from, ','.join([nd_to for nd_to in nds_to])
                                                    , is_hit
                                                    , ','.join([str(k) for k in zip(filter_nds, filter_vals)]))
                if not cnt%100:
                    fout.write(wrt_lns)
                    wrt_lns = ''
                    print('Writing %d records'%cnt)

                # search_res = [list() for nm in range(n_model)] # Find search results under each model
                # cnt_search_res = [defaultdict(int) for nm in range(n_model)]
                # col_filter_res = set() # Collabrative filtering
                # cnt_col_filter_res = defaultdict(int)
                # for nm in range(n_model):
                #     res_nd_from = model_res['src'][nm][nd_from]
                #     for dim_idx in range(n_dim):
                #         for nd in dim_index[nm]['end'][dim_idx][res_nd_from[dim_idx]]:
                #             cnt_search_res[nm][nd] += 1
                #             if cnt_search_res[nm][nd]>=filter_thres:
                #                 search_res[nm].append([nd, cnt_search_res[nm][nd]])
                #     for nd, s_cnt in search_res[nm]:
                #         cnt_col_filter_res[nd] += 1
                #         if cnt_col_filter_res[nd]>=n_model*col_prop:
                #             col_filter_res.add(nd)

                # cnt += 1
                # cand_lens.append(len(col_filter_res))
                # if col_filter_res:
                #     is_hit = False
                #     for nd_to in nds_to:
                #         if nd_to in col_filter_res:
                #             hits += 1
                #             hit_cand_lens.append(len(col_filter_res))
                #             is_hit = True
                #             break
                #     wrt_lns += '({},{}):{}:{}\n'.format(nd_from, ','.join([nd_to for nd_to in nds_to])
                #                                     , is_hit, ','.join([nd for nd in col_filter_res]))
                # if not cnt%100:
                #     fout.write(wrt_lns)
                #     wrt_lns = ''
            if cnt%100:
                fout.write(wrt_lns)
            fout.write('Hits rate: %f\n'%(hits/cnt))
