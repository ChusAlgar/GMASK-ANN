import nltk
import string
from nltk.corpus import reuters
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from collections import defaultdict
#import encoding as
import numpy as np


def isTest(fileid):
    return fileid.split("/")[0] == "test"


# Create stems from tokens in doc
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def busca_categoria(punto, info_cat):

    ind_training = info_cat['ind_training']
    cont = 0
    for list in ind_training:
        if punto >= list[0] and punto <= list[1]:
            categorias = info_cat['id_cat']
            categoria = categorias[cont]
        cont += 1

    return categoria


#####################
# FREQUENCY VECTORS #
#####################
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


def read_reuters(encoding):
    token_dict = {}
    stemmer = PorterStemmer()

    # Número total de categorías
    cats = reuters.categories()
    print("Categorias: ", cats)
    # print("Reuters tiene: ", (len(cats), cats))

    # Párrafos por categoría
    # total = len(reuters.paras())
    # total_multi = 0
    # for c in cats:
    #     lc = len(reuters.paras(categories=[c]))
    #     total_multi += lc
    #     print("%s ---- %d documents out of %d" % (c, lc, total))
    # # Estadísticas totales
    # print("Paragraphs belong to %.4f categories on average" %
    # ((total_multi * 1.0) / total))
    # print("There are %.4f paragraphs per category on average" %
    # ((total * 1.0) / len(cats)))

    # Elegimos las categorías con las que nos quedamos:
    categories = ['bop', 'cpi']  # ['earn', 'acq'] ['acq', 'alum']
                                     # ['alum', 'barley']
                                     # barley y alum son muy parecidas en tamaño
    # Estructura en la que voy almacenar el índice del primer
    # y el último documento de cada categoría
    indt = []
    cont_category = 0
    num_training = 0
    num_test = 0
    for category in categories:
        # arts_cat = reuters.fileids(categories)
        arts_cat = reuters.fileids(category)
        # arts_acq = reuters.fileids()
        # print(len(arts_cat))

        ini_training = num_training
        for art_name in arts_cat:
            if art_name.split("/")[0] == "test":
                num_test += 1
            else:
                num_training += 1
        fin_training = num_training-1
        indt.append([ini_training, fin_training])

        cont_category += 1


# print("Arts. en categoría 'acq' de training: ", num_training)
# print("Arts. en categoría 'acq' de test: ", num_test)

    if encoding == 1:
        ####################
        # TFIDF con steming#
        ####################
        # Create tokens dict from docs stems
        for file in reuters.fileids(categories):
            # Filter only training docs
            # We take each doc as the first paragraph in each document
            if not isTest(file):
                token_dict[file] = stem_tokens(reuters.paras(fileids=[file])[0][0],
                                               stemmer)

    # PRUEBA QUITAR LOS QUE NO CLASIFICA BIEN
    # Bloque de código para quitar los documentos que no clasifica bien
    # list_quita = [6, 15, 18, 29, 33, 37, 38, 39, 42, 47, 54, 55, 59, 65, 69,
    # 71]
    # dict_keys = list(token_dict.keys())
    # for contador in list_quita:
    #     print("Categorías de", dict_keys[contador], ":",
    #           reuters.categories(dict_keys[contador]))
    # for contador in list_quita:
    #     print("Texto", dict_keys[contador], ":")
    #     print(reuters.paras(fileids=[dict_keys[contador]]))
    #
    # print()
    # print()
    # list_quita = [0, 4, 10, 20, 30, 40, 50, 60]
    # dict_keys = list(token_dict.keys())
    # for contador in list_quita:
    #     print("Texto", dict_keys[contador], ":",
    #           reuters.paras(fileids=[dict_keys[contador]]))
    # print("Length of token_dict", len(token_dict))

        tf_idf = TfidfVectorizer(stop_words='english', input='content')
        # tfs será una matriz dispersa con clave (doc_id, term) y valor el tf_idf
        tfs = tf_idf.fit_transform([" ".join(l) for l in token_dict.values()])

        ####################
        # TFIDF sin steming#
        ####################
        files = [f for f in reuters.fileids(categories) if 'training' in f]
        text = [reuters.raw(fileids=[f]) for f in files]
        tf_idf = TfidfVectorizer(stop_words='english', input='content')
        tfs = tf_idf.fit_transform(text)

    else:
        ####################
        # FREQUENCY VECTOR #
        ####################
        # Con funciones de Scikit-Learn
        # Quitan los determinantes, las preposicioines y los verbos del corpus
        files = [f for f in reuters.fileids(categories) if 'training' in f]
        text = [reuters.raw(fileids=[f]) for f in files]
        vectorizer = CountVectorizer(stop_words='english')
        tfs = vectorizer.fit_transform(text)
        # print(vectorizer.get_feature_names())
        #print(tfs.toarray())
        if encoding == 3:
            ####################
            # ONE-HOT ENCODING #
            ####################
            # Con funciones de Scikit-Learn
            onehot = Binarizer()
            tfs = onehot.fit_transform(tfs)
            #print(tfs.toarray())

    id_categorias = []
    cont = 0
    for i in categories:
        id_categorias.append(cont)
        cont += 1
    # PRUEBA QUITAR LOS QUE NO CLASIFICA BIEN
    # indt=[[0,24],[25,56]]
    info_cat ={'id_cat': id_categorias, 'nom_cat': categories,
               'ind_training': indt, 'num_cat': len(categories)}
    # Should be sparse matrix object!
    # print(type(tfs))

    # First row of sparse matrix
    # for t in tfs[0]:
    #    print(t)

    # Recover words in first doc
    # print(tf_idf.inverse_transform(tfs[0]))

    return tfs, info_cat


