from keras import activations, layers
from keras import optimizers
from keras import losses
import keras.backend as K
from keras.engine.training import Model
from six import u
import tensorflow as tf
import keras


def multi_conv(input_tensor):
	first = layers.Conv2D(filters=16, kernel_size=[1, 1], padding="same")(input_tensor)
	for i in range(3):
		second = layers.Conv2D(filters=8, kernel_size=[3,3], padding="same")(input_tensor)
		second = layers.Conv2D(filters=16, kernel_size=[3,3], padding="same")(second)
		second = layers.Conv2D(filters=16, kernel_size=[1,1], padding="same")(second)
	for i in range(3):
		third = layers.Conv2D(filters=16, kernel_size=[5,5], padding="same")(input_tensor)
		third = layers.Conv2D(filters=32, kernel_size=[5,5], padding="same")(third)
		third = layers.Conv2D(filters=32, kernel_size=[1,1], padding="same")(third)
	
	out = layers.concatenate([first, second, third], axis=-1)
	return out

def SAM(input_tensor):
	max_pool = layers.Lambda(lambda x : K.max(x, axis=3, keepdims=True))(input_tensor)
	avg_pool = layers.Lambda(lambda x : K.mean(x, axis=3, keepdims=True))(input_tensor)

	max_avg_pool = layers.concatenate([max_pool, avg_pool], axis=3)

	conv = layers.Conv2D(filters=1, kernel_size=[3,3], padding="same", activation="sigmoid")(max_avg_pool)
	
	return conv

def CAM(input_tensor, reduction_ratio):
	print("--------CAM-----------")

	channel = int(input_tensor.shape[3])
	avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
	avg_pool = layers.Reshape((1,1,channel))(avg_pool)
 
	max_pool = layers.GlobalMaxPooling2D()(input_tensor)
	max_pool = layers.Reshape((1,1,channel))(max_pool)
	print(avg_pool)
	FC1 = layers.Dense(units=int(channel * reduction_ratio), 
							 activation='relu', 
							 kernel_initializer='he_normal')
	
	FC2 = layers.Dense(units=int(channel), 
							 activation='relu', 
							 kernel_initializer='he_normal')
	
	mlp_1_max = FC1(max_pool)
	mlp_2_max = FC2(mlp_1_max)
	mlp_2_max = layers.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)

	mlp_1_avg = FC1(avg_pool)
	mlp_2_avg = FC2(mlp_1_avg)
	mlp_avg = layers.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)

	channel_attention_feature = layers.add([mlp_2_avg, mlp_2_max])

	channel_attention_feature = layers.Activation("sigmoid")(channel_attention_feature)

	return layers.Multiply()([channel_attention_feature, input_tensor])

def CBAM(input_tensor, reduction_ratio=0.5):
	channel = input_tensor.shape[3]
	if channel > 1:
		channel_feature = CAM(input_tensor, reduction_ratio)
		print(channel_feature)

	else:
		channel_feature = input_tensor

	spatial_attention_feature = SAM(channel_feature)
	feature_map = layers.Multiply()([channel_feature, spatial_attention_feature])

	return layers.Add()([input_tensor, feature_map])


def MultiScale(freqs, features ,channels, faults):
	input_tensor = layers.Input([freqs, features, channels])
	out = multi_conv(input_tensor)
	
	out = CBAM(out)

	out = multi_conv(out)
	out = CBAM(out)
	
	out = layers.Flatten()(out)
	out = layers.Dense(units=1000, activation="relu")(out)
	out = layers.Dense(units=300, activation="relu")(out)
	out = layers.Dense(units=faults, activation="softmax")(out)
	model = Model(input_tensor, out, name="MultiScale")
	return model
	