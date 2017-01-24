import string
from collections import Counter
import tqdm
import nltk
import re

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
VOCABULARY_SIZE = 6500
UNK = "<UNK>"
PAD = "<PAD>"

def clean(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    for i in range(10):
        text = text.replace(str(i), " " + str(i) + " ")
    text = MATCH_MULTIPLE_SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n"))


genuine = open("data/genuine.txt").read().lower()
genuine = clean(genuine)

clickbait = open("data/clickbait.txt").read().lower()
clickbait = clean(clickbait)


words = nltk.word_tokenize(genuine) + nltk.word_tokenize(clickbait)
glove_vocabulary = open("data/vocabulary.glove.txt").read().split("\n")
counts = Counter(word for word in words if word in glove_vocabulary)

vocabulary = [PAD, UNK] + [word for word, count in counts.most_common(VOCABULARY_SIZE-2)]

def mark_unknown_words(sentence):
    return " ".join(word if word in vocabulary else UNK for word in sentence.split(" "))

genuine = [mark_unknown_words(sentence)  for sentence in tqdm.tqdm(genuine.split("\n"), desc="genuine")]
clickbait = [mark_unknown_words(sentence) for sentence in tqdm.tqdm(clickbait.split("\n"), desc="clickbait")]

open("data/vocabulary.txt", "w").write("\n".join(vocabulary))
open("data/genuine.preprocessed.txt", "w").write("\n".join(genuine))
open("data/clickbait.preprocessed.txt", "w").write("\n".join(clickbait))