from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 3]
y = [0, 25.60, 37.81, 41.76]

# x = [0, 1, 2, 3, 4]
# y = [0, 9.59, 13.52, 15.54, 15.95]
x = [1, 2, 3, 4, 5]
y1 = [0.91538459, 0.93461537, 0.93461537, 0.89615387, 0.9269231]
y2 = [0.99615383, 0.97692305, 0.98461539, 0.9826923, 0.9653846]
y3 = [0.78854167, 0.78333336, 0.78125,    0.79374999, 0.81145835]
y4 = [0.70955884, 0.6727941,  0.68897057, 0.69779414, 0.69191176]
plt.plot(x, y1, color="red", label="Sallen-key")
plt.plot(x, y2, color="green", label="Four-opamp")
plt.plot(x, y3, color="skyblue", label="leapfrog_filter")
plt.plot(x, y4, color="blue", label="ellipitic_filter")
plt.legend(loc="upper right")
plt.xlabel("five fold")
plt.ylabel("Accuracy")
plt.show()