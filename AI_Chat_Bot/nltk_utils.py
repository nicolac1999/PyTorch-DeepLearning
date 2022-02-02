# The model tries to recognize the phrase of the customer, classify it in a category and 
# give a response

# bag of words vector --> feed forward neural network --> class probabilities

# tokenization --> splitting a string into meaningful units (words,punctuation characters,numbers)
# stemming --> generate the root form of the words, crude heuristc that chops of the ends off of words

# NLP preprocessing pipeline
# tokenize
# lower+stem
# exclude punctuation characters
# bag of words

import nltk
import numpy as np

#nltk.download('punkl')

from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()

def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    '''
    sentence= ['hello','how','are','you']
    words= ['hi','hello','I', 'you','bye','thank','cool']
    bag=   [ 0 ,     1  , 0 ,   1  ,  0  ,   0   ,   0  ]
    '''
    tokenized_sentence=[stem(w) for w in tokenized_sentence]

    bag=np.zeros(len(all_words),dtype=np.float32)

    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0

    return bag

if __name__ == '__main__':

    a="How long does shipping take?"
    print(tokenization(a))


    word=['organize','organizes','organizing']
    stemmed_words=[stem(w) for w in word]
    print(stemmed_words)

    sentence= ['hello','how','are','you']
    words= ['hi','hello','I', 'you','bye','thank','cool']
    bag=bag_of_words(sentence,words)
    print(bag)


