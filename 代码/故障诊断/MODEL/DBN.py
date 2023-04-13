#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

# https://codechina.csdn.net/mirrors/myme5261314/dbn_tf/-/blob/master/dbn_tf.py
"""
This file implement a class DBN.
"""

import os
from re import S
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
print(BASE_DIR)
sys.path.append(BASE_DIR)
from sklearn import preprocessing

from dbn_tf1.RBM import RBM
from copy import deepcopy
import numpy as np
import os
import math
import tensorflow as tf

class DBN(object):
    
    """Docstring for DBN. """

    def __init__(self, sizes, opts, faults, x_train, y_train, x_test, y_test):
        """初始化参数，循环构建RBM层

        Args:
            sizes ([list]): [RBM 隐层的单元数列表]
            opts ([list]): []
            X ([None, features]): [所有的训练数据]
        """
        
        self._sizes = sizes
        self._opts = opts
        self.faults = faults
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.rbm_list = []
        
        self.features = x_train.shape[1]
        input_size = x_train.shape[1]
        self.w_list = []
        self.b_list = []
        self.init_param()
        for i, size in enumerate(self._sizes):
            self.rbm_list.append(RBM("rbm_%d".format(i), input_size, size, self._opts))
            input_size = size
            
    def init_param(self):
        input_size = self.features
        for size in self._sizes + [self.faults]:
            max_range = 4 * math.sqrt(6. / (input_size + size))
            self.w_list.append(
                np.random.uniform(
                    -max_range, max_range, [input_size, size]
                ).astype(np.float32))
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size
   
    def pre_train(self):
        """TODO: Docstring for train.
        :returns: TODO

        """
        X = self.x_train
        # 每层单独训练
        for i, rbm in enumerate(self.rbm_list):
            rbm.train(X)
            X = rbm.rbmup(X)
            # if not os.path.exists("./save/rbm{}.npz".format(i)):
                
            # else:
            #     print("./save/rbm{}has exists".format(i))

        self.load_pretrain_param()
        
    def load_pretrain_param(self):
        print("load_pre_train_param")
        assert len(self._sizes) == len(self.rbm_list)
        for i in range(len(self._sizes)):
            
            self.w_list[i] = self.rbm_list[i].w
            self.b_list[i] = self.rbm_list[i].hb
        
        
    def train(self):
        
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder(tf.float32, [None, self.features], name="train_x")
        y = tf.placeholder(tf.float32, [None, self.faults], name="train_y")
        
        for i in range(len(self.w_list)):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, _w[i])
        regularizer = tf.contrib.layers.l2_regularizer(0.001)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)
        for i in range(1, len(self._sizes)+1):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i-1], _w[i-1]) + _b[i-1], name=f"layers{i}")

        out = tf.matmul(_a[-2], _w[-1]) + _b[-1]
        logits = tf.nn.softmax(out)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
        loss += reg_term
        tf.summary.scalar("loss", loss)
        # optimizer = tf.train.MomentumOptimizer(self._opts._learning_rate, self._opts._momentum).minimize(loss)
        optimizer = tf.train.AdamOptimizer(self._opts._learning_rate).minimize(loss)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        graph = tf.get_default_graph()

        with tf.Session(graph=graph) as sess:
           
            # out = self.DBN_net()
            
            tf.summary.scalar("acc", accuracy)
            sess.run(tf.global_variables_initializer())
            
            merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        
            writer = tf.summary.FileWriter("./log/", sess.graph)

            for i in range(self._opts._epoches):
                for start, end in zip(range(0, len(self.x_train), self._opts._batchsize),
                                      range(self._opts._batchsize,
                                            len(self.x_train), self._opts._batchsize)):
                    batch_x = self.x_train[start:end]
                    batch_y = self.y_train[start:end]
                    

                    summary, train_op, train_loss, train_acc = sess.run([merged, optimizer, loss, accuracy],
                                                               feed_dict={_a[0]:batch_x, y:batch_y})
                    # all_acc = sess.run(accuracy, feed_dict={self.x:self.x_train, self.y:self.y_train})

                if (i+1) % 20 == 0:
                    test_acc, test_loss = sess.run([accuracy, loss], feed_dict={_a[0]:self.x_test, y:self.y_test})
                    print("epoch {}, loss is {:.4f}, train_acc is {:.3f}, test_loss is {:.4f} test_acc is {:.3f}".format(i+1, train_loss, train_acc, test_loss, test_acc))
                writer.add_summary(summary, i)
            writer.close()

            for i in range(len(self._sizes) + 1):
                self.w_list[i] = sess.run(_w[i])
                self.b_list[i] = sess.run(_b[i])
            
    def predict(self, x_test, y_test):
        
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder(tf.float32, [None, self.features], name="test_x")
        y = tf.placeholder(tf.float32, [None, self.faults], name="test_y")
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])
        for i in range(1, len(self._sizes)+1):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i-1], _w[i-1]) + _b[i-1])
        out = tf.matmul(_a[-2], _w[-1]) + _b[-1]
        logits = tf.nn.softmax(out)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        pred = tf.one_hot(tf.argmax(logits, 1), depth=self.faults)
        with tf.Session() as sess:
            train_pred, train_acc = sess.run([pred, accuracy], feed_dict={_a[0]:x_test, y:y_test})
        return train_pred, train_acc
    
    def final_layer_out(self, x):
        
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder(tf.float32, [None, self.features], name="test_x")
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])
        for i in range(1, len(self._sizes)+1):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i-1], _w[i-1]) + _b[i-1])
        with tf.Session() as sess:
            final_out = sess.run(_a[-2], feed_dict={_a[0]:x})
        return final_out
