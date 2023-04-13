from keras import layers
from keras import activations
from keras import losses
from keras.engine.training import Model


def DAE(sizes):
    features = sizes[0]
    classes = sizes[-1]
    input_tensor = layers.Input([features, ], name="input")
    x = layers.Dropout(0.2)(input_tensor)
    for i in range(1, len(sizes)-1):
        x = layers.Dense(units=sizes[i], activation=activations.relu)(x)
    
    out1 = layers.Dense(units=features, activation=activations.relu, name="decode")(x)
    out2 = layers.Dense(units=classes, activation=activations.softmax, name="class")(x)
    
    model = Model(input_tensor, [out1, out2])
    
    return model
    