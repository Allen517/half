# -*- coding:utf8 -*-
from __future__ import print_function

import random
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict

from utils.LogHandler import LogHandler
from utils.utils import load_train_valid_labels, read_features, batch_iter, valid_iter, write_in_file

class HALF_SP(object):

    def __init__(self, learning_rate, batch_size, neg_ratio, gamma, eta
                    , n_input, n_out, n_hidden, n_layer, type_model, is_valid
                    , device, files, log_file):
        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        self.device = device

        self.type_model = type_model

        # Parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.valid = is_valid
        self.valid_prop = .9 if self.valid else 1.
        self.valid_sample_size = 10

        self.gamma = gamma
        self.eta = eta

        self.cur_epoch = 1

        # Network Parameters
        self.n_hidden = n_hidden if type_model=='mlp' else n_input # number of neurons in hidden layer
        self.n_input = n_input # size of node embeddings
        self.n_out = n_out # hashing code
        self.n_layer = n_layer # number of layer

        # Set Train Data
        if not isinstance(files, list) and len(files)<3:
            self.logger.info('The alogrihtm needs inputs: feature-src, feature-end, identity-linkage')
            return

        # tf Graph input
        self.lookup = defaultdict(dict)
        self.look_back = defaultdict(list)
        self._read_train_dat(files) # features from source, features from end, label file
        self.valid_sample_size = min(min(self.valid_sample_size, len(self.look_back['src'])-1)
                                    , len(self.look_back['end'])-1)

        # TF Graph Building
        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)
        with tf.device(self.device):
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                self._init_weights()
                self.build_graph(type_model)
                self.build_valid_graph(type_model)
            self.sess.run(tf.global_variables_initializer())

    def _read_train_dat(self, files):
        self.F, self.lookup['src'], self.look_back['src'] \
                = read_features(files['feat-src'])
        self.G, self.lookup['end'], self.look_back['end'] = read_features(files['feat-end'])
        self.L = load_train_valid_labels(files['linkage'], self.lookup, self.valid_prop)

    def _init_weights(self):
        # Store layers weight & bias
        self.weights = dict()
        self.biases = dict()
        if self.type_model=='mlp':
            # inputs
            self.weights['h0_src'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
            self.weights['h0_end'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
            self.biases['b0_src'] = tf.Variable(tf.zeros([self.n_hidden]))
            self.biases['b0_end'] = tf.Variable(tf.zeros([self.n_hidden]))
            # hidden
            for i in range(1,self.n_layer):
                self.weights['h{}'.format(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
                self.biases['b{}'.format(i)] = tf.Variable(tf.zeros([self.n_hidden]))
        # outputs
        self.weights['out'] = tf.Variable(tf.random_normal([self.n_hidden, self.n_out]))
        self.biases['b_out'] = tf.Variable(tf.zeros([self.n_out]))

    def build_mlp_code_graph(self, inputs, tag):

        # Input layer
        layer = tf.nn.sigmoid(tf.add(tf.matmul(tf.reshape(inputs,[-1,self.n_input]), self.weights['h0_'+tag])
                                , self.biases['b0_'+tag]))
        for i in range(1,self.n_layer):
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights['h{}'.format(i)])
                                    , self.biases['b{}'.format(i)]))
        # Output fully connected layer with a neuron
        code = tf.nn.tanh(tf.matmul(layer, self.weights['out']) + self.biases['b_out'])

        return code

    def build_lin_code_graph(self, inputs, tag):

        # Output fully layer with a neuron
        code = tf.nn.tanh(tf.matmul(tf.reshape(inputs,[-1,self.n_input]), self.weights['out']) + self.biases['b_out'])

        return code

    def build_train_graph(self, src_tag, end_tag, code_graph):

        PF = code_graph(self.inputs_pos[src_tag], src_tag) # batch_size*n_out
        PG = code_graph(self.inputs_pos[end_tag], end_tag) # batch_size*n_out
        NF = tf.reshape(
                code_graph(self.inputs_neg[src_tag], src_tag)
                , [-1, self.neg_ratio, self.n_out]
                ) # batch_size*neg_ratio*n_out
        NG = tf.reshape(
                code_graph(self.inputs_neg[end_tag], end_tag)
                , [-1, self.neg_ratio, self.n_out]
                ) # batch_size*neg_ratio*n_out
        B = tf.sign(PF+PG) # batch_size*n_out
        # self.ph['B'] = tf.sign(self.ph['F']+self.ph['G']) # batch_size*n_out

        # train loss
        term1_first = tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(PF, PG),axis=1)))
        term1_second = tf.reduce_sum(tf.log(1-tf.nn.sigmoid(tf.reduce_sum(tf.multiply(NF, NG),axis=2))),axis=1)
        term1 = -tf.reduce_sum(term1_first+term1_second)
        term2 = tf.reduce_sum(tf.pow((B-PF),2))+tf.reduce_sum(tf.pow((B-PG),2))
        term3 = tf.reduce_sum(tf.pow(PF,2)+tf.reduce_sum(tf.pow(NF,2),axis=1))+tf.reduce_sum(tf.pow(PG,2)+tf.reduce_sum(tf.pow(NG,2),axis=1))
        # term1 = -tf.reduce_sum(tf.multiply(self.ph['S'], theta)-tf.log(1+tf.exp(theta)))
        # term2 = tf.reduce_sum(tf.norm(self.ph['B']-self.ph['F'],axis=1))+tf.reduce_sum(tf.norm(self.ph['B']-self.ph['G'],axis=1))
        # term3 = tf.reduce_sum(tf.norm(self.ph['F'],axis=1))+tf.reduce_sum(tf.norm(self.ph['G'],axis=1))
        self.term1 = term1
        self.term2 = term2
        self.term3 = term3

        return (term1+self.gamma*term2+self.eta*term3)/self.cur_batch_size

    def build_graph(self, type_code_graph):
        self.cur_batch_size = tf.placeholder('float32', name='batch_size')

        self.inputs_pos = {
            'src': tf.placeholder('float32', [None, self.n_input]),
            'end': tf.placeholder('float32', [None, self.n_input])
        }
        self.inputs_neg = {
            'src': tf.placeholder('float32', [None, self.neg_ratio, self.n_input]),
            'end': tf.placeholder('float32', [None, self.neg_ratio, self.n_input])
        }

        if type_code_graph=='lin':
            code_graph = self.build_lin_code_graph
        elif type_code_graph=='mlp':
            code_graph = self.build_mlp_code_graph

        self.loss = (self.build_train_graph('src', 'end', code_graph) 
                            + self.build_train_graph('end', 'src', code_graph))/2.

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def build_valid_graph(self, type_code_graph):

        # validation
        self.inputs_val = {
            'src': tf.placeholder('float32', [None, self.valid_sample_size, self.n_input]),
            'end': tf.placeholder('float32', [None, self.valid_sample_size, self.n_input])
        }

        if type_code_graph=='lin':
            code_graph = self.build_lin_code_graph
        elif type_code_graph=='mlp':
            code_graph = self.build_mlp_code_graph

        valids = {
            'src': tf.reshape(
                    code_graph(self.inputs_val['src'], 'src')
                    , [-1, self.valid_sample_size, self.n_out]
                    ), # batch_size*neg_ratio*n_out
            'end': tf.reshape(
                    code_graph(self.inputs_val['end'], 'end')
                    , [-1, self.valid_sample_size, self.n_out]
                    ) # batch_size*neg_ratio*n_out 
        }

        # self.dot_dist = tf.reduce_sum(tf.multiply(valid_f, valid_g),axis=2)
        self.hamming_dist = -tf.reduce_sum(
                                tf.clip_by_value(tf.sign(tf.multiply(valids['src'],valids['end'])),-1.,0.)
                                , axis=2
                                )

    def train_one_epoch(self):
        sum_loss = 0.0
        mrr = 0.0

        # train process
        # print 'start training...'
        batches = batch_iter(self.L, self.batch_size, self.neg_ratio\
                                        , self.lookup, 'src', 'end')

        batch_id = 0
        for batch in batches:
            # training the process from source network to end network
            pos,neg = batch
            if not len(pos['src'])==len(pos['end']) and not len(neg['src'])==len(neg['end']):
                self.logger.info('The input label file goes wrong as the file format.')
                continue
            batch_size = len(pos['src'])
            feed_dict = {
                self.inputs_pos['src']:self.F[pos['src'],:],
                self.inputs_pos['end']:self.G[pos['end'],:],
                self.inputs_neg['src']:self.F[neg['src'],:],
                self.inputs_neg['end']:self.G[neg['end'],:],
                self.cur_batch_size:batch_size
            }
            _, cur_loss = self.sess.run([self.train_op, self.loss],feed_dict)

            sum_loss += cur_loss
            batch_id += 1

        if self.valid:
            # valid process
            valid = valid_iter(self.L, self.valid_sample_size, self.lookup, 'src', 'end')
            # print valid_f,valid_g
            if not len(valid['src'])==len(valid['end']):
                self.logger.info('The input label file goes wrong as the file format.')
                return
            valid_size = len(valid['src'])
            feed_dict = {
                self.inputs_val['src']:self.F[valid['src'],:],
                self.inputs_val['end']:self.G[valid['end'],:],
            }
            # valid_dist = self.sess.run(self.dot_dist,feed_dict)
            valid_dist = self.sess.run(self.hamming_dist,feed_dict)
            for i in range(valid_size):
                fst_dist = valid_dist[i][0]
                pos = 1
                for k in range(1,len(valid_dist[i])):
                   if fst_dist>=valid_dist[i][k]:
                       pos+=1
                # print pos
                # self.logger.info('dist:{},pos:{}'.format(fst_dist,pos))
                # print valid_dist[i]
                mrr += 1./pos
            self.logger.info('Epoch={}, sum of loss={!s}, mrr={}'
                                .format(self.cur_epoch, sum_loss/(batch_id+1e-8), mrr/(valid_size+1e-8)))
        else:
            self.logger.info('Epoch={}, sum of loss={!s}'
                                .format(self.cur_epoch, sum_loss/batch_id))

        self.cur_epoch += 1

        # print(sum_loss/(batch_id+1e-8), mrr/(valid_size+1e-8))
        return sum_loss/(batch_id+1e-8), mrr/(valid_size+1e-8)

    def save_models(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        for k,v in self.weights.items():
            if self.type_model == 'lin':
                if 'out' not in k:
                    continue
            write_in_file(filename, v.eval(self.sess), k)
        for k,v in self.biases.items():
            if self.type_model == 'lin':
                if 'out' not in k:
                    continue
            write_in_file(filename, v.eval(self.sess), k)

if __name__ == '__main__':
    res_file = 'res_file'

    # SAVING_STEP = 1
    # MAF_EPOCHS = 21
    # model = DCNH(learning_rate=0.1, batch_size=4, neg_ratio=3, n_input=4, n_out=2, n_hidden=3
    #               ,files=['tmp_res.node_embeddings_src', 'tmp_res.node_embeddings_obj', 'data/test.align'])
    SAVING_STEP = 10
    MAF_EPOCHS = 20001
    model = DCNH_SP(learning_rate=0.01, batch_size=128, neg_ratio=5, n_input=256, n_out=32, n_hidden=32, n_layer=2
                    ,files=['douban_all.txt', 'weibo_all.txt', 'douban_weibo.identity.users.final.p0dot8']
                    ,log_file='DCNH_SP'
                    ,device=':/gpu:0')
    for i in range(MAF_EPOCHS):
        model.train_one_epoch()
        if i>0 and i%SAVING_STEP==0:
            model.save_models(res_file+'.epoch_'+str(i))