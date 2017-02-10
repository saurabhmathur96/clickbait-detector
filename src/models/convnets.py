from keras.models import Sequential, Model
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2

def ConvolutionalNet(vocabulary_size, embedding_dimension, input_length, embedding_weights=None):
    
    model = Sequential()
    if embedding_weights is None:
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, trainable=False))
    else:
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, weights=[embedding_weights], trainable=False))

    model.add(Convolution1D(32, 2, W_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Convolution1D(32, 2, W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Convolution1D(32, 2, W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling1D(17))
    model.add(Flatten())

    model.add(Dense(1, bias=True, W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    return model


