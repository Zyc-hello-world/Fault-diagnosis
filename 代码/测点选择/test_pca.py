import numpy as np
from sklearn.decomposition import PCA


def combine(n, k):
	res = []
	tmp = []
	def dfs(start, level, tmp):
		if n-start + 1 < level : return
		if level == 0: res.append(tmp[::])
		for i in range(start, n+1):
			tmp.append(i)
			dfs(i+1, level-1, tmp)
			tmp.pop()
	dfs(1, k, tmp)
	return res


r = combine(14, 2)

def change2Binay(res, nodes):
	pop = []
	for one in res:
		number = list(np.zeros(nodes))
		for num in one:
			number[num-1] = 1
		pop.append(number)
	return pop

pop = change2Binay(r, 14)

# print(r)
# print(pop)

res = [(p1, p2) for (p1, p2) in zip(r, pop)]
print(res)