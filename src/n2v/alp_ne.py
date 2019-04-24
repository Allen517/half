# -*- coding:utf8 -*-
from __future__ import print_function

import random
import math
import numpy as np
from collections import defaultdict
from utils.LogHandler import LogHandler
import os

class _ALP_NE(object):

    def __init__(self, graphs, lr=.001, gamma=.1, rep_size=128, batch_size=100, negative_ratio=5, table_size=1e8,
                    anchor_file=None, log_file='log', last_emb_files=dict()):

        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        self.epsilon = 1e-7
        self.table_size = table_size
        self.sigmoid_table = {}
        self.sigmoid_table_size = 1000
        self.SIGMOID_BOUND = 6

        self._init_simgoid_table()

        self._init_dicts()
        self.t = 1
        self.rep_size = rep_size
        for graph_type in ['f', 'g']:
            self.g[graph_type] = graphs[graph_type]
            self.look_up[graph_type] = self.g[graph_type].look_up_dict
            self.idx[graph_type] = 0
            self.update_dict[graph_type] = dict()
            self.update_look_back[graph_type] = list()
            self.node_size[graph_type] = self.g[graph_type].node_size
            self.embeddings[graph_type], self.h_delta[graph_type], self.m[graph_type], self.v[graph_type]\
                    = self._init_params(self.node_size[graph_type], rep_size,
                                            last_emb_files, graph_type)
            self._gen_sampling_table(graph_type)

        self.anchors = self._read_anchors(anchor_file, ',')

        self.lr = lr
        self.gamma = gamma
        self.cur_epoch = 0
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

    def _init_dicts(self):
        self.g = dict()
        self.look_up = dict()
        self.idx = dict()
        self.update_dict = dict()
        self.update_look_back = dict()
        self.node_size = dict()
        self.embeddings = dict()
        self.h_delta = dict()
        self.m = dict()
        self.v = dict()
        self.node_degree = dict()
        self.sampling_table = dict()
        self.edge_alias = dict()
        self.edge_prob = dict()

    def _init_params(self, node_size, rep_size, last_emb_file, graph_type):
        embeddings = dict()
        embeddings['node'] = np.random.normal(0,1,(node_size,rep_size))
        embeddings['content'] = np.random.normal(0,1,(node_size,rep_size))
        if last_emb_file:
            embeddings['node'] = self._init_emb_matrix(embeddings['node']\
                        , '{}.node_embeddings'.format(last_emb_file[graph_type]), graph_type)
            embeddings['content'] = self._init_emb_matrix(embeddings['content']\
                        , '{}.content_embeddings'.format(last_emb_file[graph_type]), graph_type)
        # adagrad
        h_delta = dict()
        h_delta['node'] = np.zeros((node_size,rep_size))
        h_delta['content'] = np.zeros((node_size,rep_size))
        # adam
        m = dict()
        m['node'] = np.zeros((node_size,rep_size))
        m['content'] = np.zeros((node_size,rep_size))
        v = dict()
        v['node'] = np.zeros((node_size,rep_size))
        v['content'] = np.zeros((node_size,rep_size))

        return embeddings, h_delta, m, v

    def _init_emb_matrix(self, emb, emb_file, graph_type):
        with open(emb_file, 'r') as embed_handler:
            for ln in embed_handler:
                elems = ln.strip().split()
                if len(elems)<=2:
                    continue
                emb[self.look_up[graph_type][elems[0]]] = map(float, elems[1:])
        return emb

    def _read_anchors(self, anchor_file, delimiter):
        anchors = list()
        with open(anchor_file, 'r') as anchor_handler:
            for ln in anchor_handler:
                elems = ln.strip().split(delimiter)
                anchors.append((elems[0], elems[1]))
        return anchors

    def _init_simgoid_table(self):
        for k in range(self.sigmoid_table_size):
            x = 2*self.SIGMOID_BOUND*k/self.sigmoid_table_size-self.SIGMOID_BOUND
            self.sigmoid_table[k] = 1./(1+np.exp(-x))

    def _fast_sigmoid(self, val):
        if val>self.SIGMOID_BOUND:
            return 1-self.epsilon
        elif val<-self.SIGMOID_BOUND:
            return self.epsilon
        k = int((val+self.SIGMOID_BOUND)*self.sigmoid_table_size/self.SIGMOID_BOUND/2)
        return self.sigmoid_table[k]
        # return 1./(1+np.exp(-val))

    def _format_vec(self, vec, graph_type):
        len_gap = self.idx[graph_type]-len(vec)
        if len_gap>0:
            num_col = 0
            if isinstance(vec, list):
                num_col = len(vec[0])
            else:
                num_col = vec.shape[1]
            vec = np.concatenate((vec, np.zeros((len_gap, num_col))))
            # for i in range(len_gap):
            #     vec = np.append(vec, np.zeros(vec[0].shape))
        return np.array(vec)

    def _calc_delta_vec(self, nd, delta, opt_vec, graph_type):
        if nd not in self.update_dict[graph_type]:
            cur_idx = self.idx[graph_type]
            self.update_dict[graph_type][nd] = cur_idx
            self.update_look_back[graph_type].append(nd)
            self.idx[graph_type] += 1
        else:
            cur_idx = self.update_dict[graph_type][nd]
        if cur_idx>=len(delta):
            for i in range(cur_idx-len(delta)):
                delta.append(np.zeros(opt_vec.shape))
            delta.append(opt_vec)
        else:
            delta[cur_idx] += opt_vec
        return delta

    def _update_graph_by_links(self, batch, graph_type):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        pos_h, pos_t, pos_h_v, neg_t = batch[graph_type]
        batch_size = len(pos_h)
        # print pos_h, pos_t, pos_h_v, neg_t

        embeddings = self.embeddings[graph_type]
        # order 2
        pos_u = embeddings['node'][pos_h,:]
        pos_v_c = embeddings['content'][pos_t,:]
        neg_u = embeddings['node'][pos_h_v,:]
        neg_v_c = embeddings['content'][neg_t,:]

        pos_e = np.sum(pos_u*pos_v_c, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v_c, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # temporal delta
        delta_eh = list()
        delta_c = list()

        idx = 0
        for i in range(len(pos_t)):
            u,v = pos_h[i],pos_t[i]
            delta_c = self._calc_delta_vec(v, delta_c, (sigmoid_pos_e[i]-1)*pos_u[i,:], graph_type)
            delta_eh = self._calc_delta_vec(u, delta_eh, (sigmoid_pos_e[i]-1)*pos_v_c[i,:], graph_type)
            # print 'delta_eh',delta_eh,ndDict_order
        neg_shape = neg_e.shape
        for i in range(neg_shape[0]):
            for j in range(neg_shape[1]):
                u,v = pos_h_v[i][j],neg_t[i][j]
                delta_c = self._calc_delta_vec(v, delta_c, sigmoid_neg_e[i,j]*neg_u[i,j,:], graph_type)
                delta_eh = self._calc_delta_vec(u, delta_eh, sigmoid_neg_e[i,j]*neg_v_c[i,j,:], graph_type)
                # print sigmoid_neg_e[i,j]*neg_v_c[i,j,:], type(sigmoid_neg_e[i,j]*neg_v_c[i,j,:])
                # print 'delta_eh',delta_eh,ndDict_order


        # delta x & delta codebook
        delta_eh = self._format_vec(delta_eh, graph_type)
        delta_c = self._format_vec(delta_c, graph_type)

        # print 'in update graph by links '+graph_type
        # print self.idx[graph_type], delta_eh.shape, delta_c.shape

        return delta_c/batch_size, delta_eh/batch_size

    def _cos_sim(self, vec1, vec2):
        return np.dot(vec1,vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)

    def _update_graph_by_anchor_reg(self):

        delta_eh = defaultdict(list)

        cnt = 0
        for src_nd, target_nd in self.anchors:
            if not src_nd in self.look_up['f'] or not target_nd in self.look_up['g']:
                continue
            types = ['f', 'g']
            idx = list() # 0 refers to network f, 1 refers to network g
            emb = list()
            idx.append(self.look_up['f'][src_nd])
            idx.append(self.look_up['g'][target_nd])
            emb.append(self.embeddings['f']['node'][idx[0]])
            emb.append(self.embeddings['g']['node'][idx[1]])
            for i in range(len(types)):
                delta_eh[types[i]] = self._calc_delta_vec(idx[i], delta_eh[types[i]]
                                , (self._cos_sim(emb[i], emb[1-i])*emb[i]/np.dot(emb[i], emb[i])
                                    -emb[1-i]/np.linalg.norm(emb[1-i])/np.linalg.norm(emb[i])), types[i])
            cnt += 1

        for graph_type in ['f', 'g']:
            delta_eh[graph_type] = self._format_vec(delta_eh[graph_type], graph_type)/cnt
            # print 'in update graph by anchor reg ' + graph_type
            # print self.idx[graph_type], delta_eh[graph_type].shape

        return delta_eh

    def _mat_add(self, mat1, mat2):
        # print '****mat add****'
        # print mat1, mat2
        len_gap = len(mat1)-len(mat2)
        # print len_gap
        if len_gap>0:
            for i in range(len_gap):
                mat2 = np.vstack((mat2, np.zeros(mat2[0,:].shape)))
                # print mat2
        else:
            for i in range(-len_gap):
                mat1 = np.vstack((mat1, np.zeros(mat1[0,:].shape)))
        #         print mat1
        # print len(mat1), len(mat2)
        return mat1+mat2

    def get_graph_loss(self, batch, graph_type):
        pos_h, pos_t, pos_h_v, neg_t = batch[graph_type]

        embeddings = self.embeddings[graph_type]
        # order 2
        pos_u = embeddings['node'][pos_h,:]
        pos_v_c = embeddings['content'][pos_t,:]
        neg_u = embeddings['node'][pos_h_v,:]
        neg_v_c = embeddings['content'][neg_t,:]

        pos_e = np.sum(pos_u*pos_v_c, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v_c, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_anchor_reg_loss(self):

        cos_sim_list = list()

        for src_nd, target_nd in self.anchors:
            if not src_nd in self.look_up['f'] or not target_nd in self.look_up['g']:
                continue
            src_idx = self.look_up['f'][src_nd]
            target_idx = self.look_up['g'][target_nd]
            
            cos_sim_list.append(self._cos_sim(self.embeddings['f']['node'][src_idx]
                                    , self.embeddings['g']['node'][target_idx]))

        return -np.mean(cos_sim_list)

    def update_vec(self, h_delta, delta, embeddings, len_delta, t, graph_type):
        update_look_back = self.update_look_back[graph_type]
        h_delta[update_look_back[:len_delta],:] += delta**2
        # print 'original embedding:',embeddings[self.update_look_back[cal_type][:len_delta]]
        embeddings[update_look_back[:len_delta],:] -= \
                        self.lr/np.sqrt(h_delta[update_look_back[:len_delta],:])*delta
        # print 'delta:',delta
        # print 'h_delta:',h_delta[self.update_look_back[cal_type][:len_delta]]
        # print 'embeddings:',embeddings[self.update_look_back[cal_type][:len_delta]]
        # print 'lmd_rda:',elem_lbd
        return h_delta, embeddings

    def update_vec_by_adam(self, m, v, delta, embeddings, len_delta, t, graph_type):
        self.beta1 = .9
        self.beta2 = .999
        update_look_back = self.update_look_back[graph_type]
        m[update_look_back[:len_delta],:] = \
            self.beta1*m[update_look_back[:len_delta],:]+(1-self.beta1)*delta
        v[update_look_back[:len_delta],:] = \
            self.beta2*v[update_look_back[:len_delta],:]+(1-self.beta2)*(delta**2)
        m_ = m[update_look_back[:len_delta],:]/(1-self.beta1**t)
        v_ = v[update_look_back[:len_delta],:]/(1-self.beta2**t)

        embeddings[update_look_back[:len_delta],:] -= self.lr*m_/(np.sqrt(v_)+self.epsilon)

        return m,v,embeddings

    def train_one_epoch(self, opt_type):
        DISPLAY_EPOCH=100

        def batch_init():
            for graph_type in ['f', 'g']:
                self.idx[graph_type] = 0
                self.update_look_back[graph_type] = list()
                self.update_dict[graph_type] = dict()

        batches = self.batch_iter()
        last_batch_loss = 1e8
        stop_cnt = 0
        for batch in batches:
            batch_loss = .0
            batch_init()
            delta_eh_anchor_reg = self._update_graph_by_anchor_reg()
            for graph_type in ['f', 'g']:
                # init
                h_delta = self.h_delta[graph_type]
                embeddings = self.embeddings[graph_type]
                m = self.m[graph_type]
                v = self.v[graph_type]
                # end
                delta_c, delta_eh = self._update_graph_by_links(batch, graph_type)
                delta_eh_anchor_reg[graph_type] = self._format_vec(delta_eh_anchor_reg[graph_type], graph_type)
                # print 'in train one epoch'
                # print self.idx[graph_type], delta_eh_anchor_reg[graph_type].shape, delta_eh.shape
                len_delta = len(delta_eh)
                # print 'order2, nd'
                if opt_type=='adagrad':
                    h_delta['node'], embeddings['node'] = \
                                        self.update_vec(h_delta['node']
                                                    , delta_eh+self.gamma*delta_eh_anchor_reg[graph_type]
                                                    , embeddings['node'], len_delta, self.t, graph_type)
                if opt_type=='adam':
                    m['node'], self.v['node'], embeddings['node'] = \
                                    self.update_vec_by_adam(m['node'], v['node']
                                                    , delta_eh+self.gamma*delta_eh_anchor_reg[graph_type]
                                                    , embeddings['node'], len_delta, self.t, graph_type)
                len_content = len(delta_c)
                # print 'order2, content'
                if opt_type=='adagrad':
                    h_delta['content'], embeddings['content'] = \
                                        self.update_vec(h_delta['content'], delta_c
                                                    , embeddings['content'], len_content, self.t, graph_type)
                if opt_type=='adam':
                    m['content'], v['content'], embeddings['content'] = \
                                    self.update_vec_by_adam(m['content'], v['content'], delta_c
                                                    , embeddings['content'], len_content, self.t, graph_type)
                if (self.t-1)%DISPLAY_EPOCH==0:
                    batch_loss += self.get_graph_loss(batch, graph_type)+self.gamma*self.get_anchor_reg_loss()
            if (self.t-1)%DISPLAY_EPOCH==0:
                self.logger.info('Finish processing batch {} and loss:{}'.format(self.t-1, batch_loss))
                if batch_loss<last_batch_loss:
                    last_batch_loss = batch_loss
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                if stop_cnt>=2:
                    break
            self.t += 1
        self.cur_epoch += 1

    def get_random_node_pairs(self, i, shuffle_indices, edges, edge_set, numNodes, graph_type):
        # balance the appearance of edges according to edge_prob
        edge_prob = self.edge_prob[graph_type]
        edge_alias = self.edge_alias[graph_type]
        sampling_table = self.sampling_table[graph_type]
        if i>=len(shuffle_indices):
            i = np.random.randint(len(shuffle_indices))
        if not random.random() < edge_prob[shuffle_indices[i]]:
            shuffle_indices[i] = edge_alias[shuffle_indices[i]]
        cur_h = edges[shuffle_indices[i]][0]
        head = cur_h*numNodes
        cur_t = edges[shuffle_indices[i]][1]
        cur_h_v = []
        cur_neg_t = []
        for j in range(self.negative_ratio):
            rn = sampling_table[random.randint(0, self.table_size-1)]
            while head+rn in edge_set or cur_h == rn or rn in cur_neg_t:
                rn = sampling_table[random.randint(0, self.table_size-1)]
            cur_h_v.append(cur_h)
            cur_neg_t.append(rn)
        return cur_h, cur_t, cur_h_v, cur_neg_t

    def batch_iter(self):

        data_size = 0
        for graph_type in ['f', 'g']:
            net_size = self.g[graph_type].G.size()
            if net_size > data_size:
                data_size = net_size

        shuffle_indices = dict()
        for graph_type in ['f', 'g']:
            net_size = self.g[graph_type].G.size()
            shuffle_indices[graph_type] = np.random.permutation(np.arange(net_size))
            while net_size<data_size:
                shuffle_indices[graph_type] = np.append(shuffle_indices[graph_type], shuffle_indices[graph_type][:data_size-net_size])
                net_size = len(shuffle_indices[graph_type])

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            ret = dict()

            for graph_type in ['f', 'g']:
                numNodes = self.node_size[graph_type]
                look_up = self.look_up[graph_type]
                g = self.g[graph_type]
                edges = [(look_up[x[0]], look_up[x[1]]) for x in g.G.edges()]
                edge_set = set([x[0]*numNodes+x[1] for x in edges])
                pos_h = []
                pos_t = []
                pos_h_v = []
                neg_t = []
                for i in range(start_index, end_index):
                    cur_h, cur_t, cur_h_v, cur_neg_t\
                        = self.get_random_node_pairs(i, shuffle_indices[graph_type], edges, edge_set, numNodes, graph_type)
                    pos_h.append(cur_h)
                    pos_t.append(cur_t)
                    pos_h_v.append(cur_h_v)
                    neg_t.append(cur_neg_t)
                ret[graph_type] = (pos_h, pos_t, pos_h_v, neg_t)

            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)

            yield ret

    def _gen_sampling_table(self, graph_type):
        table_size = self.table_size
        power = 0.75

        print("Pre-procesing for non-uniform negative sampling in {}!".format(graph_type))
        numNodes = self.node_size[graph_type]
        g = self.g[graph_type]

        node_degree = np.zeros(numNodes) # out degree
        look_up = g.look_up_dict
        for edge in g.G.edges():
            node_degree[look_up[edge[0]]] += g.G[edge[0]][edge[1]]['weight']

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        p = 0
        i = 0
        sampling_table = np.zeros(int(table_size), dtype=np.uint32)
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1

        data_size = g.G.size()
        edge_alias = np.zeros(data_size, dtype=np.int32)
        edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([g.G[edge[0]][edge[1]]["weight"] for edge in g.G.edges()])
        norm_prob = [g.G[edge[0]][edge[1]]["weight"]*data_size/total_sum for edge in g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            edge_prob[cur_small_block] = norm_prob[cur_small_block]
            edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            edge_prob[small_block[num_small_block]] = 1

        self.node_degree[graph_type] = node_degree
        self.sampling_table[graph_type] = sampling_table
        self.edge_alias[graph_type] = edge_alias
        self.edge_prob[graph_type] = edge_prob

    def get_one_embeddings(self, embeddings, graph_type):
        vectors = dict()
        look_back = self.g[graph_type].look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors

    def get_vectors(self):
        ret = defaultdict(dict)
        content_embeddings = defaultdict(dict)

        for graph_type in ['f', 'g']:
            node_embeddings=self.get_one_embeddings(self.embeddings[graph_type]['node'], graph_type)
            ret[graph_type]['node_embeddings']=node_embeddings

            content_embeddings=self.get_one_embeddings(self.embeddings[graph_type]['content'], graph_type)
            ret[graph_type]['content_embeddings']=content_embeddings

        return ret

class ALP_NE(object):

    def __init__(self, graphs, lr=.001, gamma=.1, rep_size=128, batch_size=1000, epoch=10, saving_epoch=1,
                    neg_ratio=5, table_size=1e8, outfile='test', anchor_file=None,
                    last_emb_files=dict(), log_file='log'):
        SAVING_EPOCH=saving_epoch
        # paramter initialization
        self.vectors = {}
        self.rep_size = rep_size
        # training
        self.model = _ALP_NE(graphs, lr=lr, gamma=gamma, rep_size=rep_size
                            , batch_size=batch_size, negative_ratio=neg_ratio
                            , table_size=table_size, log_file=log_file
                            , anchor_file=anchor_file
                            , last_emb_files=last_emb_files)

        opt_type = 'adam'
        for i in range(1,epoch+1):
            self.model.logger.info('Start to training in epoch {}'.format(i))
            self.model.train_one_epoch(opt_type)
            if i%SAVING_EPOCH==0:
                print('Saving results...')
                self.get_embeddings()
                self.save_embeddings('{}.epoch{}'.format(outfile,i))
            self.model.logger.info('End of training in epoch {}'.format(i))
        print('Saving results...')
        self.get_embeddings()
        self.save_embeddings('{}.final'.format(outfile))

    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = self.model.get_vectors()

    def save_embeddings(self, outfile):
        vectors = self.vectors
        for graph_type in vectors.keys():
            for c in vectors[graph_type].keys():
                if 'node_embeddings' in c or 'content_embeddings' in c:
                    # outfile-[node_embeddings/content-embeddings]-[src/obj]
                    fout = open('{}.{}.{}'.format(outfile,graph_type,c), 'w') 
                    node_num = len(vectors[graph_type][c].keys())
                    fout.write("{} {}\n".format(node_num, self.rep_size))
                    for node, vec in vectors[graph_type][c].items():
                        fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
                    fout.close()
