import os
import numpy as np
import nltk
from keras.models import Sequential, load_model
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GRU, LSTM
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2


SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30
MODEL_FILE = "models/detector.h5"

embedding_weights = np.load("models/embeddings.npy")
vocabulary = open("data/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))

def words_to_indices(words):
    return [inverse_vocabulary[word] for word in words]

clickbait = open("data/clickbait.preprocessed.txt").read().split("\n")
clickbait = sequence.pad_sequences([words_to_indices(sentence.split()) for sentence in clickbait], maxlen=SEQUENCE_LENGTH)

genuine = open("data/genuine.preprocessed.txt").read().split("\n")
genuine = sequence.pad_sequences([words_to_indices(sentence.split()) for sentence in genuine], maxlen=SEQUENCE_LENGTH)

X = np.concatenate([clickbait, genuine], axis=0)
y = np.array([[1] * clickbait.shape[0] + [0] * genuine.shape[0]], dtype=np.int32).T

if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    model.layers[0].trainable = False
else:
    model = Sequential()
    model.add(Embedding(len(vocabulary), EMBEDDING_DIMENSION, weights=[embedding_weights], input_length=SEQUENCE_LENGTH, trainable=False))
    model.add(BatchNormalization())
    model.add(Convolution1D(16, 2))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Convolution1D(16, 2))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(8))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1, bias=True))
    model.add(Activation("sigmoid"))

optimizer = RMSprop()
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
model.fit(X, y, validation_split=.2, batch_size=32, nb_epoch=25, shuffle=True)
model.layers[0].trainable = True
model.save(MODEL_FILE)

"""
model.add(LSTM(64, W_regularizer=l2(0.01), U_regularizer=l2(0.005), b_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Convolution1D(32, 2))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Convolution1D(32, 2))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Convolution1D(32, 2))
model.add(Activation("relu"))
model.add(MaxPooling1D(3))
model.add(Flatten())
"""