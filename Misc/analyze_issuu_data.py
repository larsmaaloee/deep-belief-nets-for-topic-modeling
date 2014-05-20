
import numpy as np
from numpy import *
import numpy.random as nr
import os
#from nltk.corpus import stopwords
import re
import string
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool, Queue
import multiprocessing
global wordpunct_tokenize
global EnglishStemmer
global stopwords

class AnalyzeIssuuData(object):
    def __init__(self, path, white_lst_path):
        self.dict = {}
        self.words = []
        self.frequencies = []
        self.path = path

        print 'Analysing Issuu data for ',path.split('/')[-1]
        self.white_lst = open(white_lst_path).read().replace(" ","").split("\n")
        self.generate_word_count_dict()
        self.__output_dict()

    def generate_word_count_dict(self):
        """
        Generate a word count dictionary that keeps the word
        count for documents in a specific path/group.
        """

        global wordpunct_tokenize
        global EnglishStemmer
        global stopwords

        # Import nltk
        from nltk.tokenize import wordpunct_tokenize as wt
        from nltk.stem.porter import PorterStemmer as ES
        from nltk.corpus import stopwords as sw
        wordpunct_tokenize = wt
        EnglishStemmer = ES
        stopwords = sw



        docs = os.listdir(self.path)
        docs = [os.path.join(self.path,d) for d in docs]

        print "Stemming docs."
        stemmer = EnglishStemmer()
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        p = Pool(6)
        p.map_async(stem_doc,[(queue,docs[i],self.white_lst,stemmer) for i in xrange(len(docs))])
        p.close()
        p.join()

        for i in xrange(len(docs)):
            tmp_dict = queue.get()
            for k in tmp_dict.keys():
                try:
                    self.dict[k] += tmp_dict[k]
                except KeyError:
                    self.dict[k] = tmp_dict[k]


            if i % 10 == 0:
                print "Processed "+str(i)+" of "+str(len(docs))


        # Compute the average.
        #for k in self.dict.keys():
        #    self.dict[k] /= len(docs)

        self.frequencies = sorted(self.dict.values())
        self.words = [x for (y,x) in sorted(zip(self.dict.values(),self.dict.keys()))]




    def __output_dict(self):
        print 'Outputting statistics to output folder.'
        f = open('output/statistics_'+self.path.split('/')[-1]+'.txt','wb')

        for i in range(len(self.words)):
            word = self.words[i]
            try:
                f.write(word+': '+str(self.frequencies[i])+"\n")
            except UnicodeEncodeError:
                continue
        f.write('Number of words: '+str(len(self.words)))
        f.close()



    def plot_barchart(self):
        plt.bar(range(len(self.frequencies)), self.frequencies, align='center')
        plt.xticks(range(len(self.words)), self.words())
        plt.show()



def stem_doc(args):
    queue, p, white_lst,stemmer = args
    d = open(p,'rb').read()
    doc = [stemmer.stem(token.lower()) for token in wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), '', d.decode(encoding = 'UTF-8',errors = 'ignore'))) if token.lower() not in stopwords.words('english')]# Stem, lowercase, substitute all punctuations, remove stopwords.
    tmp_dict = dict(Counter(doc))
    for k in tmp_dict.keys():
        if not k.encode('utf-8') in white_lst:
            tmp_dict.pop(k)

    queue.put(tmp_dict)

