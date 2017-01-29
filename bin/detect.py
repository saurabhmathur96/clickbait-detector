from keras.models import load_model
from keras.preprocessing import sequence
import sys
import string 
import re

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
SEQUENCE_LENGTH = 20

UNK = "<UNK>"
PAD = "<PAD>"

model = load_model("models/detector.h5")


vocabulary = open("data/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))

def words_to_indices(words):
    return [inverse_vocabulary.get(word, inverse_vocabulary[UNK]) for word in words]


def clean(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    for i in range(10):
        text = text.replace(str(i), " " + str(i) + " ")
    text = MATCH_MULTIPLE_SPACES.sub(" ", text)
    return text

headline = sys.argv[1].encode("ascii", "ignore")
inputs = sequence.pad_sequences([words_to_indices(clean(headline).split())], maxlen=SEQUENCE_LENGTH)
clickbaitiness = model.predict(inputs)[0, 0]
print ("headline is {0} % clickbaity".format(round(clickbaitiness * 100, 2)))