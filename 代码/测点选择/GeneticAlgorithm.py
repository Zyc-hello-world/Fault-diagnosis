#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from IPython import display


# In[2]:


class GA():
    
    # input:
    #     nums: 种群的数量
    #     max_iter: 最大迭代次数
    #     DNA_SIZE is binary bit size, None is auto
    #     cross_rate: 交叉概率
    #     mutation: 变异概率（全体变异概率），为适应度差的后几个个体生成高的变异概率。
    def __init__(self, pop_size, max_iter, func, DNA_SIZE, 
                alias_matrix, fault_num,
                cross_rate=0.8, mutation=0.003):
        
        
        #生成随机种群
        nums = np.random.randint(0, 2**DNA_SIZE, pop_size)
        if DNA_SIZE == None:
            DNA_SIZE = 1
        self.DNA_SIZE = DNA_SIZE

        #POP_SIZE为进化的种群数
        self.pop_size = pop_size
        POP = np.zeros((pop_size, DNA_SIZE))
        #用python自带的格式化转化为前面空0的二进制字符串，然后拆分成列表
        for i in range(pop_size):
            
            POP[i] = [int(k) for k in ('{0:0'+str(DNA_SIZE)+'b}').format(nums[i])]
        
        self.POP = POP
        print(type(POP), type(self.POP))
        self.new_pop = POP
        self.max_iter = max_iter
        #用于后面重置（reset）
        
        self.copy_POP = POP.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        self.fitness = []
        self.elitist = {"DNA" : "", "fitness" : 0, "node_num" : self.DNA_SIZE, "FI" : 0}
        self.alias_matrix = alias_matrix
        self.fault_num = fault_num
        self.FI = []
    #save args对象保留参数：

    #        POP_SIZE             种群大小
    #        POP                  编码后的种群[[1,0,1,...],[1,1,0,...],...]
    #                             一维元素是个体，二维元素是各个DNA[1,0,1,0]，
    #        copy_POP             复制的种群，用于重置
    #        cross_rate           染色体交换概率
    #        mutation             基因突变概率
    #        func                 适应度函数
    #        elitist              存储最优个体
    #        fitness              当前代的适应度函数
    #        max_iter             最大迭代次数

    #将编码后的DNA翻译回来（解码）
    def translateDNA(self):
        pass
    #得到适应度

    def pop2number(self, people):
        """
        将二进制编码的个体转换为测点的数字集合
        people: 二进制编码的个体
        return：测点集合
        """
        #从左到右编号
        number = []
        for i in range(len(people)):
            if people[i] == 1:
                number.append(i)
        return number

    def get_fitness(self):
        """
        yield种群计算每个个体的适应度函数
        return： 每个个体的适应度组成的array
        """

        result = []
        FI = []
        for p in self.POP:
            number = self.pop2number(p)
            fi_list, s, degree = self.func(np.copy(self.alias_matrix), number, self.fault_num) 
            FI.append(fi_list)
            fit = len(fi_list) + 1 - degree / s
            result.append(fit)

        # a = int(input("yes"))
        self.FI = np.array(FI)
        return np.array(result)
    
#自然选择（轮盘赌）
    def select(self):
        """
        选择出个体返回下标
        """
        probability = [f / np.sum(self.fitness)  for f in self.fitness]
        i = 0
        t = np.random.rand(1)
        s = 0
        for i, p in enumerate(probability):
            s = s + p
            if s > t:
                return i
        return len(probability) - 1

#染色体交叉
    def crossover(self, individual1, individual2):
        """
        允许出现重复交叉的可能
        将t位后的二进制进行交叉
        return:交叉后的两个个体
        """
        
        t = np.random.randint(1, self.DNA_SIZE - 1)   # 随机选择一点（单点交叉）
        
        (l1, l2) = (individual1[:t], individual2[:t])   # & 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
        (r1, r2) = (individual1[t:], individual2[t:])
        individual1 = np.hstack((l1, r2))
        individual2 = np.hstack((l2, r1))

        return (individual1, individual2)
                

#基因变异
    def mutate(self, individual):
        """
        将第t位的二进制编码进行变异
        """
        p = np.random.rand()
        if p < self.mutation:
            t = np.random.randint (0, self.DNA_SIZE - 1)
            if individual[t] == '0':
                individual[t] = 1
            elif individual[t] == '1':
                individual[t] = 0
        return individual
    
    def save_best_indv(self):
        """
        保存最优个体???,有待商榷，如果当前最优的个体适应度小，但是其测点数量很少的话，是否要选。
        """
        
        ind = np.argmax(self.fitness)

        a = np.copy(self.POP[ind])
        number = self.pop2number(a)
        # print(a, number, self.fi_list[ind], self.FI[ind])
        # print("最优*** 测点编号：{}, 隔离列表：{}, FI:{}".format(self.pop2number(np.copy(self.POP[ind]))),self.fi_list[ind], self.FI[ind])
        if self.elitist["fitness"] < self.fitness[ind]:
            


            self.elitist["DNA"] = np.copy(self.POP[ind])
            self.elitist["fitness"] = np.copy(self.fitness[ind])
            
            self.elitist["node_num"] = np.sum(self.elitist["DNA"])
            self.elitist["FI"] = self.FI[ind]
        
        
#进化
    def evolution(self):
        """
        
        """
        self.fitness = self.get_fitness()
        i = 0
        while True:
            
            indv1_index = self.select()
            indv2_index = self.select()
            
        
            (new_indv1, new_indv2) = self.crossover(np.copy(self.POP[indv1_index]), np.copy(self.POP[indv2_index]))
            
            new_indv1 = self.mutate(new_indv1)
            new_indv2 = self.mutate(new_indv2)
            self.new_pop[i] = new_indv1
            self.new_pop[i+1] = new_indv2
            
            i = i + 2
            if i >= self.pop_size:
                break
                
        self.save_best_indv()
        
        self.POP = self.new_pop
        
#
    def change_X(self):
        """
        把POP的二进制转化成测点的坐标
        例[0,1,1,0,1]转化成[1,2,4, -1]，-1表示类别
        """
        X = []
        for one in self.POP:
            for two in one:
                x = []
                l = len(two)
                for i in range(l):
                    if two[i] == 1:
                        x.append(l-i-1)
                x.append(-1)
            X.append(x)
        return X

#打印当前状态日志
    def log(self):
        pass

#一维变量作图

    def run(self):
        """
        运行遗传算法
        """
        
        for i in range(self.max_iter):
            self.evolution()
            fit_loc = np.argmax(self.fitness)
            fi_loc = self.FI[fit_loc]
            number = self.pop2number(self.POP[fit_loc])
            print ("iter={}, max_fit = {}, ave_fit={}, DNA = {}, number = {}".format(
                i, max (self.fitness), sum (self.fitness) / self.pop_size, self.POP[fit_loc], number))
        
    def plot_in_jupyter_1d(self, iter_time=200):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()
        for _ in range(iter_time):
            plt.cla()
            
            x = self.change_X()
        
            translate_x = self.translateDNA().reshape(self.POP_SIZE)
            
            acc = self.func(x)
            
            plt.scatter(translate_x, acc, s=200, lw=0, c='red', alpha=0.5)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())

            self.evolution()
            
    def test(self):
         for _ in range(1):
            self.evolution()


    def print_best_pop(self):
        GAresult = []
        best_fit = self.elitist["fitness"]
        best_num = self.elitist["node_num"]
        best_FI = self.elitist["FI"]
        stps = []
        for i, val in enumerate(self.elitist["DNA"]):
            if val == 1:
                stps.append(i)
        print(self.elitist["DNA"])
        print("适应度为：{},隔离故障为：{} 测点数量为：{}, 测点为：{}".format(best_fit, best_FI, best_num, stps))
        GAresult.append(best_fit)
        GAresult.append(best_num)
        GAresult.append(stps)
        return GAresult




