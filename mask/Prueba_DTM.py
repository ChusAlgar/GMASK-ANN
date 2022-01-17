import shorttext
import numpy as np
import pandas as pd
import spacy
spacy.load('en')
#import en_core_web_sm
#nlp = en_core_web_sm.load()

usprez = shorttext.data.inaugural()
#print(usprez)
docids = sorted(usprez.keys()) # ['1789-Washington', '1793-Washington', '1797-Adams'...]
#print('docids: ', docids)
usprez = [' '.join(usprez[docid])for docid in docids] # [discurso, discurso, ...]
preprocess = shorttext.utils.standard_text_preprocessor_1()
corpus = [preprocess(address).split(' ') for address in usprez]
'''print(corpus[0])
print(len(corpus[0]))
print(len(set(corpus[0])))'''
print(type(corpus))
print(type(docids))
matriz = shorttext.utils.DocumentTermMatrix(corpus, docids=docids)
ocurrences = matriz.get_token_occurences('god')
print(ocurrences)
#print(matriz)