from keras import activations, layers
from keras import optimizers, losses
import keras.backend as K
from keras.engine.training import Model
from six import u
import keras
from MODEL.MyNet import CBAM, SAM, CAM

def multi_conv(input_tensor, ch1, ch3_1, ch3_3, ch5_1, ch5_5, ch):
    first = layers.Conv1D(filters=ch1, kernel_size=1, padding="same", strides=1)(input_tensor)
#first = layers.Activation("relu")(first)
    second = layers.Conv1D(filters=ch3_1, kernel_size=1, padding="same", strides=1)(input_tensor)
    second = layers.Conv1D(filters=ch3_3, kernel_size=3, padding="same", strides=1)(second)
#second = layers.Activation("relu")(second)
#second = layers.Conv1D(filters=16, kernel_size=1, padding="same")(second)

    third = layers.Conv1D(filters=ch5_1, kernel_size=1, padding="same", strides=1)(input_tensor)
    third = layers.Conv1D(filters=ch5_5, kernel_size=5, padding="same", strides=1)(third)
#third = layers.Activation("relu")(third)
#third = layers.Conv1D(filters=32, kernel_size=1, padding="same")(third)
    four = layers.MaxPool1D(pool_size=3, padding="same",strides=1)(input_tensor)
    four = layers.Conv1D(filters=ch, kernel_size=1, padding="same")(four)
#four = layers.Activation("relu")(four)
    out = layers.concatenate([first, second, third, four], axis=-1)
    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    return out

def block(input_tensor, conv_filter, conv_kernel, pool_size=2, strides=2):
    out = layers.Conv1D(filters=conv_filter, kernel_size=conv_kernel, strides=strides,padding="same")(input_tensor)
    out = layers.MaxPool1D(pool_size=pool_size, padding="same")(out)
    out = layers.BatchNormalization()(out)
#out = layers.Activation("relu")(out)
    return out
def MultiScale(features,channels, faults):
    input_tensor = layers.Input([features, channels])
    first = layers.MaxPooling1D(pool_size=4)(input_tensor)
    first = CAM(first)
    out = block(input_tensor, 32, 7, strides=1)
    out = block(out, 64, 5, strides=1)
    incep = multi_conv(out,16,16,32,8,16,16)
#out = layers.AveragePooling1D(pool_size=2)(out)
#out = CBAM(out)
    incep = multi_conv(incep,32,32,64,16,32,32)
    incep = multi_conv(incep,32,48,96,24,64,64)
    final = layers.concatenate([first, out, incep])
#out = layers.MaxPooling1D(pool_size=2)(out)
#out = CBAM(out)
#out = layers.GlobalAveragePooling1D()(out)
    final = layers.Flatten()(final)
    final = layers.Dropout(0.5)(final)
#out = layers.Dense(units=1000, activation="relu")(out)
    final = layers.Dense(units=faults, activation="softmax")(final)
    model = Model(input_tensor, final, name="testmodel1")
    return model
    
