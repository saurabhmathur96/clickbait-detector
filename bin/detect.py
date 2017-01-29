from keras.models import load_model
from keras.preprocessing import sequence
import sys
import string 
import re

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
SEQUENCE_LENGTH = 20

UNK = "<UNK>"
PAD = "<PAD>"



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

class Predictor (object):
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict (self, headline):
        headline = headline.encode("ascii", "ignore")
        inputs = sequence.pad_sequences([words_to_indices(clean(headline).lower().split())], maxlen=SEQUENCE_LENGTH)
        clickbaitiness = self.model.predict(inputs)[0, 0]
        return clickbaitiness
predictor = Predictor("models/detector.h5")
if __name__ == "__main__":
    print ("headline is {0} % clickbaity".format(round(predictor.predict(sys.argv[1]) * 100, 2)))