import os
import numpy as np
import nltk
from keras.models import Sequential, load_model
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GRU, LSTM
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import train_test_split


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
p = np.random.permutation(y.shape[0])
X = X[p]
y = y[p]

X_train, X_test, y_train, y_test =  train_test_split(X, y, stratify=y)


if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    model.layers[0].trainable = False
else:
    model = Sequential()
    model.add(Embedding(len(vocabulary), EMBEDDING_DIMENSION, weights=[embedding_weights], input_length=SEQUENCE_LENGTH, trainable=False))
    
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


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, nb_epoch=20, shuffle=True, callbacks=[EarlyStopping(monitor="val_loss", patience=3)])
model.save(MODEL_FILE)

model.layers[0].trainable = True
for i in range(1, len(model.layers)):
    model.layers[i].trainable = False

# Finetune Embedding Layer
model.compile(loss="binary_crossentropy", optimizer=SGD(lr=.000001), metrics=["acc"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, nb_epoch=20, shuffle=True, callbacks=[EarlyStopping(monitor="val_loss", patience=3)])

for i in range(0, len(model.layers)):
    model.layers[i].trainable = True
    
model.save(MODEL_FILE)

import h5py
f = h5py.File(MODEL_FILE,"r+")
del f["optimizer_weights"]
f.close()

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