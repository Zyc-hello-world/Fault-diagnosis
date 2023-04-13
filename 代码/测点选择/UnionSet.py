#!/usr/bin/env python
# coding: utf-8

# # 路径压缩并查集

# In[1]:


class Unionset:
    def __init__(self, num):
        self.num = num
        self.parent = [i for i in range(num)]
    
    def find_parent(self, x):
        if self.parent[x] == x:
            return x
        else:
            return self.find_parent(self.parent[x])
        
    def join(self, x, y):
        p_x = self.find_parent(x)
        p_y = self.find_parent(y)
        self.parent[p_y] = p_x

