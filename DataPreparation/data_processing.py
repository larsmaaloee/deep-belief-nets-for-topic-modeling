__author__ = 'larsmaaloee'

import os
import re
import string
from random import shuffle
from multiprocessing import Pool, Process, cpu_count

from numpy import *

import serialization as s
from time import time
from itertools import chain
import env_paths
import stopwords


global wordpunct_tokenize
global EnglishStemmer


class DataProcessing:
    def __init__(self, paths, words_count=2000, batchsize=100, trainingset_size=None, trainingset_attributes=None,
                 acceptance_lst_path=None):

        """
        Initialize dataprocessing parameters.

        @param paths: Root paths to the documents.
        @param words_count: Number of words in the bag of words. Should be the same as number of input units in DBN.
        @param batchsize: The size of a single batch. One can set this as the length of the dataset to avoid batches.
        @param trainingset_size: The proportion of docs from the given path that should be devoted to be trainingset.
        @param trainingset_attributes: If the BOWs for training has been generated. The attr. should be given to the test set BOWs.
        @param acceptance_lst_path: The path to a txt file of white list words.
        """
        if trainingset_attributes == None:
            self.training = True
        else:
            self.training = False

        # If the acceptance list is given.
        if acceptance_lst_path == None:
            self.acceptance_lst = None
        else:
            self.acceptance_lst = open(acceptance_lst_path).read().replace(" ", "").split("\n")

        self.max_words_matrix = words_count
        self.trainingset_attributes = trainingset_attributes
        self.trainingset_size = trainingset_size
        self.paths = paths
        self.batchsize = batchsize


    def generate_bows(self):
        """
        Run through all steps of the dataprocessing to generate the BOWs for a training set and/or a testset.

        1. Take all serialized stemmed documents and assign them into batches. Each batch should represent an
        equal number of docs from a category, except the last batch.
        2. Calculate the number of words to extract an attributes list corresponding to the X (word_count) most used words.
        3. Generate the BOWs for all batches.

        The BOWs will be saved in an output folder of the project root.
        """
        print 'Data Processing Started'
        timer = time()
        completed = self.__read_docs_from_filesystem()
        if not completed:
            print 'Dataprocessing ended with an error.'
            return
        print 'Time ', time() - timer

        print 'Filtering Words'
        timer = time()
        # Add all text of docs as a tokenized list
        if self.trainingset_attributes == None:
            attributes = self.__set_attributes()
        else:
            attributes = self.trainingset_attributes
            s.dump(attributes, open(env_paths.get_attributes_path(self.training), "wb"))
        print 'Time ', time() - timer

        print 'Generate bag of words matrix'
        timer = time()
        # Generate a dictionary for lookup of the words
        index_lookup = dict(zip(attributes, range(len(attributes))))
        # Generate word matrix
        self.__generate_word_matrix(index_lookup)
        print 'Time ', time() - timer


    def __read_docs_from_filesystem(self):
        """
        Read all docs and assign them to batches, so that each doc category is represented equally across batches.

        """
        docs_names = []
        docs_names_split = []
        class_indices = []
        class_indices_split = []
        class_names = []
        batches = []

        print 'Generating class indices and docs names list.'
        doc_count = 0
        for folder in self.paths:
            docs_names_split.append([])
            class_indices_split.append([])
            class_names.append(folder.split('/')[len(folder.split('/')) - 1])
            if self.trainingset_size == None:  # If data processing should be done on all data in the specified folders.
                docs = os.listdir(folder)
            elif not self.trainingset_size == None and self.trainingset_attributes == None:  # If data processing should be done on parts of the docs in the specified folders - for training and testing purposes.
                docs = os.listdir(folder)[:int(len(os.listdir(folder)) * self.trainingset_size)]
            else:  # If data processing should be done on a test set.
                docs = os.listdir(folder)[int(len(os.listdir(folder)) * self.trainingset_size):]
            for doc in docs:
                if doc.endswith('.p'):
                    # Append the name of the document to the list containing document names.
                    docs_names_split[-1].append(folder + '/' + doc)
                    class_indices_split[-1].append(len(class_names) - 1)
                    doc_count += 1

        if len(docs_names_split) == 0:  # Check if docs have been stemmed.
            print 'Documents have not been stemmed. Please stem documents in order to create bag of words matrices.'
            return 0

        # Ensure that batches contain an even amount of docs from each category.
        print 'Arranging the documents.'
        if doc_count < self.batchsize:
            print 'Number of documents must be greater than batchsize. Please revise the batchsize.'
            return 0
        number_of_batches = doc_count / self.batchsize
        number_of_classes = len(self.paths)
        batches_collected_class_indices = []
        batches_collected_docs_names = []

        # Calculate fraction of category in each batch.
        d = {}
        for i in range(len(class_indices_split)):
            d[i] = float(len(class_indices_split[i])) / number_of_batches

        count = 0
        for i in range(number_of_batches):
            batch_class_indices = []
            batch_docs_names = []
            d_tmp = array([int(v) for v in d.values()])
            while True:
                if (len(batch_class_indices) == self.batchsize) and (not doc_count - count < self.batchsize) or (
                            count == doc_count):
                    break
                if len(d_tmp[d_tmp > 0]) == 0:
                    break
                for j in range(number_of_classes):
                    if (len(batch_class_indices) == self.batchsize) and (not doc_count - count < self.batchsize) or (
                                count == doc_count):
                        break
                    if len(class_indices_split[j]) > 0 and d_tmp[j] != 0:
                        batch_class_indices.append(class_indices_split[j].pop(0))
                        batch_docs_names.append(docs_names_split[j].pop(0))
                        d_tmp[j] -= 1
                        count += 1
            batches_collected_class_indices.append(batch_class_indices)
            batches_collected_docs_names.append(batch_docs_names)

        for i in range(number_of_batches):
            bsize = self.batchsize if i < number_of_batches - 1 else self.batchsize + (doc_count % self.batchsize)
            batch_class_indices = batches_collected_class_indices[i]
            batch_docs_names = batches_collected_docs_names[i]
            if len(batch_class_indices) < bsize:
                while True:
                    if len(batch_class_indices) == bsize: break
                    for j in range(number_of_classes):
                        if len(batch_class_indices) == bsize: break
                        if len(class_indices_split[j]) > 0:
                            batch_class_indices.append(class_indices_split[j].pop(0))
                            batch_docs_names.append(docs_names_split[j].pop(0))


            # Shuffle the batch
            batch_class_indices_shuf = []
            batch_docs_names_shuf = []
            index_shuf = range(len(batch_class_indices))
            shuffle(index_shuf)
            for k in index_shuf:
                batch_class_indices_shuf.append(batch_class_indices[k])
                batch_docs_names_shuf.append(batch_docs_names[k])

            # Append batch to full lists
            class_indices += batch_class_indices_shuf
            docs_names += batch_docs_names_shuf

        print 'Reading and saving docs from file system'
        count = 0
        class_indices_batch = []
        docs_names_batch = []
        docs_list = []
        for i in xrange(len(class_indices)):
            if not count == 0 and (
                        count % self.batchsize) == 0:  # Save the batch if batchsize is reached or if the last document has been read.
                if not (len(class_indices) - count) < self.batchsize:
                    print 'Read ', str(count), ' of ', len(class_indices)
                    self.__save_batch_loading_docs(count, docs_list, docs_names_batch, class_indices_batch)
                    batches.append(count)
                    # Reset the lists
                    docs_list = []
                    docs_names_batch = []
                    class_indices_batch = []

            d = s.load(open(docs_names[i], 'rb'))
            docs_list.append(d)
            docs_names_batch.append(docs_names[i])
            class_indices_batch.append(class_indices[i])
            count += 1

        # Save the remaining docs
        if len(docs_list) > 0:
            print 'Read ', str(count), ' of ', len(class_indices)
            self.__save_batch_loading_docs(count, docs_list, docs_names_batch, class_indices_batch)
            batches.append(count)

        s.dump(class_names, open(env_paths.get_class_names_path(self.training), "wb"))
        s.dump(batches, open(env_paths.get_batches_path(self.training), "wb"))
        return 1


    def __set_attributes(self):
        """
        Set the attributes containing of a list of words of all attributes
        in the bag of words matrix.

        @return: The generated list of words acting as attributes for the BOWs.
        """
        batches = s.load(open(env_paths.get_batches_path(self.training), "rb"))
        length = len(batches)
        attributes = []
        processed = 1
        for batch in batches:
            docs_list = s.load(open(env_paths.get_doc_list_path(self.training, batch), "rb"))
            tmp_attributes = list(
                set(sorted(list(chain(*docs_list)))))  # Retrieve the each word of the docs list in a sorted list
            attributes += tmp_attributes
            attributes = list(
                set(sorted(attributes)))  # Sort the attributes list so that there is no 2 occurrences of the same word.
            if not self.acceptance_lst == None: attributes = list(
                set(attributes).intersection(self.acceptance_lst))  # Only consider words in the acceptance list.
            print 'Processed attribute ' + str(processed) + ' of ' + str(length) + ' batches'
            processed += 1

        # Find attributes of the most common words.
        d = dict.fromkeys(attributes)
        processed = 1
        for batch in batches:
            docs_list = s.load(open(env_paths.get_doc_list_path(self.training, batch), "rb"))
            words = list(list(chain(*docs_list)))
            for w in words:
                try:
                    if d[w] == None:
                        d[w] = 1
                    else:
                        d[w] += 1
                except KeyError:
                    continue
            print 'Processed summing ' + str(processed) + ' of ' + str(length) + ' batches'
            processed += 1
        sorted_att = sorted(d.items(), key=lambda x: x[1])
        sorted_att = sorted_att[len(sorted_att) - self.max_words_matrix:]
        attributes = [elem[0] for elem in sorted_att]

        # Serialize attributes
        s.dump(attributes, open(env_paths.get_attributes_path(self.training), "wb"))
        return attributes

    def __save_batch_loading_docs(self, batch_number, docs_list, docs_names, class_indices):
        """
        Save batches for the document loading process in the initialization phase. This is done due to vast sizes
        of data - lack of memory.

        @param batch_number: Representing the number of documents in the batch.
        @param docs_list: List containing a string for each document in the batch.
        @param docs_names: List containing the names of each document in the same order as the docs_list.
        @param class_indices: List containing which class/folder each document belongs to.
        """
        # Serialize all relevant variables
        s.dump(docs_list, open(env_paths.get_doc_list_path(self.training, batch_number), "wb"))
        s.dump(docs_names, open(env_paths.get_doc_names_path(self.training, batch_number), "wb"))
        s.dump(class_indices, open(env_paths.get_class_indices_path(self.training, batch_number), "wb"))


    def __generate_word_matrix(self, index_lookup):
        """
        Generate a BOW matrix with rows, columns corresponding to documents, words respectively.

        @param index_lookup: A dictionary with keys for the attributes. In order to know which colounm should be incremented in word_matrix.
        """
        batches = s.load(open(env_paths.get_batches_path(self.training), "rb"))
        length = len(batches)
        processed = 1
        for batch in batches:
            docs_list = s.load(open(env_paths.get_doc_list_path(self.training, batch), "rb"))
            bag_of_words_matrix = zeros([len(docs_list), len(index_lookup)])
            row = 0
            for doc in docs_list:
                for token in doc:
                    try:  # If word is not found in the dictionary
                        col = index_lookup[token]
                        bag_of_words_matrix[row, col] += 1
                    except KeyError:
                        continue
                row += 1
            # Serialize bag of words
            s.dump(bag_of_words_matrix.tolist(), open(env_paths.get_bow_matrix_path(self.training, batch), "wb"))
            print 'Processed ' + str(processed) + ' of ' + str(length) + ' batches'
            processed += 1


def stem_docs_parallel(paths):
    """
    Stem all documents from the given path names

    @param paths: paths to the documents.
    """

    # Following code is the initialisation of a dummy process in order to allow numpy and multiprocessing to run
    # properly on OSX.
    stem_process = Process(target=__stem_docs, args=(paths,))
    stem_process.start()
    stem_process.join()


def __stem_docs(paths):
    # Import nltk tools
    global wordpunct_tokenize
    global EnglishStemmer
    from nltk.tokenize import wordpunct_tokenize as wt
    # from nltk.stem.snowball import EnglishStemmer
    from nltk.stem.porter import PorterStemmer as ES
    wordpunct_tokenize = wt
    EnglishStemmer = ES

    print 'Stemming documents in parallel.'
    docs = []
    for folder in paths:
        d = os.listdir(folder)
        docs += [os.path.join(x, y) for (x, y) in zip([folder for _ in range(len(d))], d)]

    tmp_docs = []
    for doc in docs:
        if "." in doc.split(os.path.sep)[-1]:
            if doc.endswith(".txt"): tmp_docs.append(doc)
        else:
            rename = doc + ".txt"
            os.rename(doc, rename)
            tmp_docs.append(rename)
    docs = tmp_docs
    p = Pool(cpu_count())
    p.map_async(__stem_doc, zip([i for i in range(len(docs))], docs))
    p.close()
    p.join()


def __stem_doc(doc_details):
    idx, doc = doc_details
    if idx % 100 == 0:
        print "Processed doc " + str(idx)
    if doc.endswith('.txt'):
        d = open(doc).read()
        stemmer = EnglishStemmer()  # This method only works for english documents.
        # Stem, lowercase, substitute all punctuations, remove stopwords.
        attribute_names = [stemmer.stem(token.lower()) for token in wordpunct_tokenize(
            re.sub('[%s]' % re.escape(string.punctuation), '', d.decode(encoding='UTF-8', errors='ignore'))) if
                           token.lower() not in stopwords.get_stopwords()]
        s.dump(attribute_names, open(doc.replace(".txt", ".p"), "wb"))


def get_bag_of_words_matrix(batch, training=True):
    """
    Retrieve the bag of words matrix for a batch.
    
    @param batch: the number of the batch.
    """
    return array(s.load(open(env_paths.get_bow_matrix_path(training, int(batch)), "rb")))


def get_batch_list(training=True):
    """
    Retrieve the list containing the batch numbers.

    @param training: is this the training set or the test set.
    """
    return s.load(open(env_paths.get_batches_path(training), "rb"))


def get_document_name(row, batch, training=True):
    """
    The name of the document corresponding to a row
    in a batch.

    @param row: row in the bag of words matrix in batch.
    @param batch: the number of the batch.
    @param training: is this the training set or the test set.
    """
    return s.load(open(env_paths.get_doc_names_path(training, batch), "rb"))[row]


def get_document_names(batch, training=True):
    """
    Get document names.
    
    @param batch: the number of the batch.
    @param training: is this the training set or the test set.
    """
    names = s.load(open(env_paths.get_doc_names_path(training, batch), "rb"))
    return names


def get_all_document_names(training=True):
    batches = get_batch_list(training)
    doc_names_collected = []

    for batch in batches:
        doc_names_collected += list(s.load(open(env_paths.get_doc_names_path(training, int(batch)), "rb")))

    return doc_names_collected


def get_class_indices(batch, training=True):
    """
    Get all class indices of the documents in a batch.
    
    @param batch: the number of the batch.
    @param training: is this the training set or the test set.
    """

    indices = s.load(env_paths.get_class_indices_path(training, batch), "rb")
    return indices


def get_document_class(row, batch, training=True):
    """
    The class of a document corresponding to a row
    in a batch.
    
    @param row: row in the bag of words matrix in batch.
    @param batch: the number of the batch.
    @param training: is this the training set or the test set.
    """
    class_indices_for_batch = s.load(open(env_paths.get_class_indices_path(training, batch), "rb"))
    class_names_for_batch = s.load(open(env_paths.get_class_names_path(training), "rb"))
    return class_names_for_batch[class_indices_for_batch[row]]


def get_attributes(training=True):
    """
    Get the attributes.
    
    @param training: is this the training set or the test set.
    """
    return s.load(open(env_paths.get_attributes_path(training), "rb"))


def get_all_class_indices(training=True):
    """
    Get all class indices for all batches in one list.

    @param training: is this the training set or the test set.
    """
    batches = get_batch_list(training)
    indices_collected = []

    for batch in batches:
        indices_collected += list(s.load(open(env_paths.get_class_indices_path(training, int(batch)), "rb")))

    return indices_collected


def get_all_class_names():
    """
    Get all class names for training set.
    """

    return s.load(open(env_paths.get_class_names_path(train=True), 'rb'))


def get_class_names_for_class_indices(class_indices):
    """
    Get all class names for the class indices.
    @param class_indices: The class indices for whom class names must be retrieved.
    """
    class_names = get_all_class_names()

    class_names_filtered = []
    for idx in class_indices:
        class_names_filtered.append(class_names[idx])
    return class_names_filtered


def stem_acceptance_list(path):
    """
    Stem the acceptance list given by the path. This should be done before data preparation for that specific list.

    @param path: The path to the acceptance list.
    """
    global EnglishStemmer
    from nltk.stem.porter import PorterStemmer as ES

    EnglishStemmer = ES

    acceptance_lst = open(path).read().replace(" ", "").split("\n")
    stemmer = EnglishStemmer()
    acceptance_lst_stemmed = []
    for word in acceptance_lst:
        acceptance_lst_stemmed.append(stemmer.stem(word.lower()))

    f = open(env_paths.get_acceptance_lst_path(), 'w')
    for w in acceptance_lst_stemmed[:-1]:
        f.write(w + "\n")
    f.write(acceptance_lst_stemmed[-1])
    f.close()

