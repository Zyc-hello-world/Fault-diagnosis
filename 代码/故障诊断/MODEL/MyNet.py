import sys
import os
import keras

from keras.layers.recurrent import LSTM
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from keras import activations, layers
from keras import optimizers
from keras import losses
import keras.backend as K
from keras.engine.training import Model
import tensorflow as tf

from DataProcessing.Signal_preprocessing import get_wpt_data

def conv1D_block(input_tensor, size, out_channels, name_base):
	"""卷积块，包含了一维的卷积，dropout, maxpool, batch_norm

	Args:
		input_tensor ([tensor]): [features, channels]
		size ([int]): [卷积核尺寸]
		out_channels ([int]): [输出通道数]
		

	Returns:
		[type]: [description]
	"""
	x = layers.Conv1D(out_channels, kernel_size=size, name=name_base+"_conv")(input_tensor)
	x = layers.MaxPool1D(pool_size=2, name=name_base+"_maxpool")(x)
	x = layers.BatchNormalization(name=name_base+"_batchnorm")(x)
	x = layers.Activation("relu")(x)
	
	return x

def SAM(input_tensor):
    max_pool = layers.Lambda(lambda x : K.max(x, axis=2, keepdims=True))(input_tensor)
    avg_pool = layers.Lambda(lambda x : K.mean(x, axis=2, keepdims=True))(input_tensor)

    max_avg_pool = layers.concatenate([max_pool, avg_pool], axis=2)

    conv = layers.Conv1D(filters=1, kernel_size=3, padding="same", activation="sigmoid")(max_avg_pool)
    
    return conv

def FM(input_tensor):
    get_wide_data()
    pass
	
def CBAM(input_tensor, reduction_ratio=0.5):
    channel_feature = CAM(input_tensor, reduction_ratio)

    spatial_attention_feature = SAM(channel_feature)

    feature_map = layers.Multiply()([channel_feature, spatial_attention_feature])
    
    return layers.Add()([input_tensor, feature_map])

def CAM(input_tensor, reduction_ratio=0.5):
	channel = int(input_tensor.shape[2])

	ave_pool = layers.GlobalAveragePooling1D()(input_tensor)
	max_pool = layers.GlobalMaxPooling1D()(input_tensor)
	FC1 = layers.Dense(units=int(channel * reduction_ratio), 
							 activation='relu', 
							 kernel_initializer='he_normal')
	
	FC2 = layers.Dense(units=int(channel), 
							 activation='relu', 
							 kernel_initializer='he_normal')
	
	mlp_max = FC1(max_pool)
	mlp_max = FC2(mlp_max)

	mlp_max = layers.Reshape(target_shape=(1, int(channel)))(mlp_max)

	mlp_avg = FC1(ave_pool)
	mlp_avg = FC2(mlp_avg)
	mlp_avg = layers.Reshape(target_shape=(1, int(channel)))(mlp_avg)

	channel_attention_feature = layers.add([mlp_avg, mlp_max])

	channel_attention_feature = layers.Activation("relu")(channel_attention_feature)

	return layers.Multiply()([channel_attention_feature, input_tensor])

class Wide():
	def __init__(self) -> None:
		pass
	def Fullconnection(self, input_tensor):
		FC = layers.Dense(units=30, activation="relu", name="wpt_FC1")(input_tensor)
		# FC2 = layers.Dense(units=15, activation="relu", name="wpt_FC2")(FC1)
		return FC

	def FM(self):
		pass

	def FFM(self):
		pass

	def LSTM(self, input_tensor):

		nb_lstm_outputs = 30  #神经元个数
		nb_time_steps = 96  #时间序列长度
		nb_input_vector = 1 #每个特征向量化的长度
		l1 = layers.LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector))(input_tensor)
		return l1

	def wide_fun(self):
		if self.type == "FM":
			return self.FM()
		elif self.type == "FFM":
			return self.FFM()
		elif self.type == "LSTM":
			return self.LSTM()	
		else:
			return self.Fullconnection()

def MyNet(conv_features, wpt_features, channels, classes, params):
	conv_input_tensor = layers.Input([conv_features, channels])
	wide_input_dict = {"lstm":layers.Input([wpt_features, 1]),
						"others":layers.Input([wpt_features])}
	
	wide_input_tensor = wide_input_dict["others"]
	print("输入的维度为")
	print(f"conv_shape {conv_input_tensor}, wide_shape {wide_input_tensor}")
	kernel_sizes = params["kernel_sizes"]
	channel_list = params["channels"]
	dense_list = params["dense_list"]
	x = conv_input_tensor
	for i in range(len(kernel_sizes)):
		x = conv1D_block(x, 
                       size=kernel_sizes[i], 
                       out_channels=channel_list[i], 
                       name_base="conv{}".format(i))
		
	x = CBAM(x)
	flatten = layers.Flatten()(x)
	wide = Wide()
	# wide_out = wide.LSTM(wide_input_tensor)
	wide_out = wide.Fullconnection(wide_input_tensor)
	print(f"flatten:{flatten}, wide_out:{wide_out}")
	dense_input = layers.concatenate([flatten, wide_out])

	for i, number in enumerate(dense_list):
		FC = layers.Dense(units=number, activation="relu", name=f"FC{i}".format(i+1))(dense_input)
		FC = layers.Dropout(0.3)(FC)
		dense_input = FC

	out = layers.Dense(units=classes, activation="softmax", name="out")(FC)
	model = Model([conv_input_tensor, wide_input_tensor], out, name="MyNet")
	return model
	
	
	
	
	
	
	
