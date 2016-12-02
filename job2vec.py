from pprint import pprint
import pandas as pd
from gensim import corpora
import gensim
import logging
import re
import multiprocessing
from collections import defaultdict
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path_raw = 'joblistings.merged.parsed.unique.grpbyyear.2010-2015.01.tsv'
path_train = 'data/va_train_clean.csv'
path_test = 'data/va_test_clean.csv'

def get_corpus(fname=path_raw):
    with open(fname) as f:
        lines = iter(f)
        lines.next()
        for line in lines:
            yield line

class CorpusFriendly(object):
    def __init__(self,fname, topK=None):
        self.topK = topK
        self.fname = fname

    def __iter__(self):
        k = 1
        corpus_memory_friendly = get_corpus(self.fname)
        for data in corpus_memory_friendly:
            data = gensim.utils.to_unicode(data).split(',')
            #print data
            words = data[2].split()
            label = [data[0],data[1],str(k)]
            yield gensim.models.doc2vec.TaggedDocument(words, label)

            if self.topK:
                if k >= self.topK:
                    break
                k += 1


if __name__ == '__main__':
    corpus_train = CorpusFriendly(fname=path_train)
    model = gensim.models.doc2vec.Doc2Vec(size=2000, min_count=3, iter=10, window=8, workers=6)
    print "Building vocab..."
    model.build_vocab(corpus_train)
    print "Training..."
    model.train(corpus_train)
    print "Training...Done"
    print "Saving model"
    try:
        model.save("models/gensim_doc2vec_3_labels")
    except:
        model.save("gensim_doc2vec_3_labels")
    print "Saveded!"
