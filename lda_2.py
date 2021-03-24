import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords

import numpy as np
# np.random.seed(2018)

nltk.download('wordnet')

import re

import pandas as pd
import numpy as np

from pprint import pprint
import spacy
from spacy.lang.fr import French
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger

# %matplotlib inline
from matplotlib import pyplot as plt
import logging
logging.basicConfig(format= '%(asctime)s : %(levelname)s : %(message)s', level = logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UnicodeWarning)
import pandas as pd

papers = pd.read_csv('/home/lea/Téléchargements/article_dataset.csv')
#papers = pd.read_csv('./Article_data/article_dataset.csv')

# Je souhaite afficher le début du dataframe
#print(papers.head())

# On retire la ponctuation
papers['Text_processed'] = \
papers['Text'].map(lambda x : re.sub('[,\.!?]', '', x))

# On transforme toutes les lettres en minuscule
papers['Text_processed'] = \
papers['Text_processed'].map(lambda x : x.lower())


@Language.factory('french_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer(after_melt=True, default=True)

@Language.factory('melt_tagger')
def create_melt_tagger(nlp, name):
    return POSTagger()

nlp = spacy.load('fr_core_news_md')
nlp.add_pipe('melt_tagger', after='parser')
nlp.add_pipe('french_lemmatizer', after='melt_tagger')

def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

# nlp = spacy.load("fr_core_news_sm")
# nlp = spacy.load('fr_core_news_md')
nlp.tokenizer = custom_tokenizer(nlp)
nlp.tokenizer.token_match = French.Defaults.token_match

data_tokenize = []
for article in papers['Text_processed']:
    article_nlp = nlp(article)
    data_sentence = []
    for token in article_nlp:
        word_lem = token.lemma_
        data_sentence.append(str(word_lem))
    data_tokenize.append(data_sentence)

data_tokenize_list = []
for article in data_tokenize:
    doc_list = []
    for doc in article:
#         if str(doc) not in stopwords_list:
        doc_list.append(str(doc))
        no_integers = [x for x in doc_list if not (x.isdigit()
                                         or x[0] == '-' and x[1:].isdigit())]
    data_tokenize_list.append(no_integers)

morestopwords = ['dela', 'd\'', '-','\n\n', '\n','il','-t','-il', 'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', \
                 'dimanche', 'ans', 'faire', 'pay','sûr', '%', '-ils', ':', '>', '«', ' ', '»', "'", '(', ')','«il', \
                  'ni(ait', 'ent', '-elle', 'trop', '’', '–', 'fois','re', 'déjà','-là', '—','jamais','année','|','[',\
                 ']', 'pourcent','"', 'aller', '\u2009', 'an', 'nouveau', 'heure', 'million', 'jusqu’','heure', 'semaine',\
                 'nouveau', 'dernier', 'taux', 'semaine', 'grand', 'fin', '#', 'the', 'fin', 'pouvoir', \
                 'sur', 'contre', 'où', 'chez', 'd\'une', 'autant', 'ont-ils', 'il', 'un', 'on', 'également', 'qu\'il',\
                 'est-il', 'est', 'qu', 'n\'est', 'c\'est', 's\'est', 'l\'est', 'qu\'on', 'lorsqu', 'lorsqu\'on', 'presque', \
                 'jusque-là', 'qu\'un', 'qu\'une', 'lorsqu\'il', 'qu\'avec', 'qu\'en', 'd\'où', 'presqu\'un', 'jusqu\'au', \
                 'jusqu\'en', 'jusqu\'aux', 'd\'un', 'd\'une', 'qu\'elle','janvier', 'février', 'mars', 'avril', 'mai', 'juin',\
                 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre', 'jour', 'pays', 'france', '...', 'toutefois', 'passer', 'devoir',\
                 'continuer', 'donner', 'beaucoup']

stopwords_list = stopwords.words('french')
stopwords_list.extend(morestopwords)
stopwords_list.extend(list(fr_stop))

# remove stop words from tokens
data_stop = []
for article in data_tokenize_list:
    stopped_tokens = [i for i in article if not i in stopwords_list]
    data_stop.append(stopped_tokens)

# Create Dictionary
id2word = corpora.Dictionary(data_stop)
# Create Corpus
texts = data_stop
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

#LDA training

from pprint import pprint
# number of topics
num_topics = 20
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics, passes=20)
# Print the Keyword in the 20 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]



