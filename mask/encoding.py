from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
import nltk
import string
from collections import defaultdict

from nltk.corpus import reuters


corpus = ["The elephant sneezed at the sight of potatoes.",
           "Bats and bats can see via echolocation. See the bat sight sneeze!",
           "Wondering, she opened the door to the studio."]

#################################
# FREQUENCY VECTORS SCIKIT-LEARN#
#################################
# Con funciones de Scikit-Learn
# Quitan los determinantes, las preposicioines y los verbos del corpus
print("FREQUENCY VECTORS SCIKIT-LEARN")
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())


#########################
# FREQUENCY VECTORS NLTK#
#########################
# Con funciones programadas del libro
# No se quitan ni los determinantes, ni las preposiciones ni los verbos
def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue
        yield stem.stem(token)


def vectorize(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features


print("FREQUENCY VECTORS NLTK")
vectors = map(vectorize, corpus)
print(list(vectors))


########################
# ONE-HOT ENCODING NLTK#
########################
# Con funciones programadas del libro
# No se quitan ni los determinantes, ni las preposiciones ni los verbos
def vectorize_one_hot(doc):
    return {
        token: True
        for token in tokenize(doc)
    }


print("# ONE-HOT ENCODING NLTK#")
corpus = ["The elephant sneezed at the sight of potatoes.",
           "Bats and bats can see via echolocation. See the bat sight sneeze!",
           "Wondering, she opened the door to the studio."]
vectors = map(vectorize_one_hot, corpus)
print(list(vectors))


################################
# ONE-HOT ENCODING SCIKIT-LEARN#
################################
# Con funciones de Scikit-Learn
# Quitan los determinantes, las preposicioines y los verbos del corpus
print("ONE-HOT ENCODING SCIKIT-LEARN")
freq = CountVectorizer(stop_words='english')
X = freq.fit_transform(corpus)
onehot = Binarizer()
X = onehot.fit_transform(X.toarray())
print(X)
