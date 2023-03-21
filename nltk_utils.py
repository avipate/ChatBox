# Importing required libraries
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


# Tokenize the sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Applying Stemming
def stem(word):
    return  stemmer.stem(word.lower())


# bag of words
def bag_of_words(tokenized_sentence, all_words):
    pass


words = ['Organize', 'organizes', 'organizing']
stemmed_words = [stem(w) for w in words]
print(stemmed_words)