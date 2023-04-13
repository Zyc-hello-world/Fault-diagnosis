import tensorflow as tf
a = 0
print(tf.__version__)
if tf.test.gpu_device_name():
	print("Device is {}".format(tf.test.gpu_device_name()))
else:
	print("please install GPU ")
