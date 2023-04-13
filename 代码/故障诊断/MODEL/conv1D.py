from keras import layers
from keras import activations
from keras import losses
from keras.engine.training import Model
from keras import regularizers



def block(input_tensor, size, out_channels, dropprob):
    """卷积块，包含了一维的卷积，dropout, maxpool, batch_norm

    Args:
        input_tensor ([tensor]): [features, channels]
        size ([int]): [卷积核尺寸]
        out_channels ([int]): [输出通道数]
        dropprob ([float]): [丢弃率]

    Returns:
        [type]: [description]
    """

    x = layers.Conv1D(out_channels, 
                      kernel_size=size)(input_tensor)
    # x = layers.Dropout(dropprob)(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x
    

def Conv1DNet(features, channels, classes, params):
    """[summary]

    Args:
        features ([int]): [description]
        channels ([int]): [输入通道数]
        classes ([int]): [description]
        params ([2-Dlist]): [[kernel, out_channel], [...]]
        droprob (float, optional): [description]. Defaults to 0.2.

    Returns:
        [type]: [description]
    """
    input_tensor = layers.Input([features, channels], name="input")
    kernel_list = params["kernel"]   
    channel_list = params["channel"] 
    FC_list = params["FC"]
    droprob = params["droporb"]
    x = input_tensor
    for i in range(len(kernel_list)):
        conv_block1 = block(x, kernel_list[i], channel_list[i], droprob[i])
        x = conv_block1    
    flatten = layers.Flatten()(x)
    
    x = flatten
    for i in range(len(FC_list)):
        FC = layers.Dense(units=FC_list[i], 
                          activation=activations.relu, 
                        name=f"FC{i}")(x)
        x = FC

    out = layers.Dense(units=classes, activation=activations.softmax, name="class")(x)
    
    model = Model(input_tensor, out, name="conv1DNet")
    
    return model
