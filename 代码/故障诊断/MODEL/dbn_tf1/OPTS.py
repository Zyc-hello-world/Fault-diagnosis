#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
This file implement a Opt class for storing some hyper-parameter of DL.
"""


class DLOption(object):

    """Docstring for DLOption. """

    def __init__(self, epoches, learning_rate, batchsize, momentum):
        """TODO: to be defined1.

        :epoches: 迭代次数
        :learning_rate: TODO
        :batchsize: TODO
        :momentum: TODO
        :penaltyL2: TODO
        :dropout: TODO
        :dropoutProb: TODO

        """
        self._pre_train_epoches = 200
        self._epoches = epoches
        self._learning_rate = learning_rate
        self._batchsize = batchsize
        self._momentum = momentum
        # self._penaltyL2 = penaltyL2
        # self._dropoutProb = dropoutProb

