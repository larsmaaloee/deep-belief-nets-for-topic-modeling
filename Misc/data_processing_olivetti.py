__author__ = 'larsmaaloee'

from sklearn.datasets import fetch_olivetti_faces
from numpy import *
import cPickle as m
import os

def data_processing_olivetti():
    """
    Python function for importing the Olivetti data set.
    """
    dataset = fetch_olivetti_faces()
    faces = dataset.data
    n_samles, n_features = faces.shape
    class_indices = dataset['target']

    train_set = []
    train_class_indices = []
    train_batches = []
    test_set = []
    test_class_indices = []
    test_batches = []

    curr_idx_count = 0
    batch_count_train = 0
    batch_count_test = 0
    for i in range(len(class_indices)):
        if curr_idx_count <= 6:
            train_set.append(faces[i].reshape((1,len(faces[i]))))
            train_class_indices.append(array([class_indices[i]]))
            train_batches.append(batch_count_train)
            batch_count_train += 1
        elif curr_idx_count <=9:
            test_set.append(faces[i].reshape((1,len(faces[i]))))
            test_class_indices.append(array([class_indices[i]]))
            test_batches.append(batch_count_test)
            batch_count_test += 1
        if curr_idx_count == 9:
            curr_idx_count = -1

        curr_idx_count += 1



    train_path = "output/train/bag_of_words"
    os.makedirs(train_path)
    m.dump(array(train_batches),open(os.path.join(train_path,"batches.p"),"wb"))
    for i in range(len(train_set)):
        m.dump(train_set[i],open(os.path.join(train_path,"bow_batch_"+str(train_batches[i]))+".p","wb"))
        m.dump(train_class_indices[i],open(os.path.join(train_path,"class_indices_batch_"+str(train_batches[i]))+".p","wb"))


    test_path = "output/test/bag_of_words"
    os.makedirs(test_path)
    m.dump(array(test_batches),open(os.path.join(test_path,"batches.p"),"wb"))
    for i in range(len(test_set)):
        m.dump(test_set[i],open(os.path.join(test_path,"bow_batch_"+str(test_batches[i]))+".p","wb"))
        m.dump(test_class_indices[i],open(os.path.join(test_path,"class_indices_batch_"+str(test_batches[i]))+".p","wb"))


def collect_batches():
    train_path = "output/train/bag_of_words"
    test_path = "output/test/bag_of_words"

    train_batches = m.load(open(os.path.join(train_path,"batches.p"),"rb"))
    collected_batches = []
    collected_class_indices = []
    batches = []
    count = 0
    for b in train_batches:
        if count % 100 == 0 and not count == 0:
            print "Processed %d"%count
            m.dump(collected_batches,open(os.path.join(train_path,"bow_batch_")+str(count)+".p","wb"))
            m.dump(collected_class_indices,open(os.path.join(train_path,"class_indices_batch_")+str(count)+".p","wb"))
            batches.append(count)
            collected_batches = []
            collected_class_indices = []
        data = m.load(open(os.path.join(train_path,"bow_batch_")+str(b)+".p","rb"))
        collected_batches.append(data[0])
        class_indices = m.load(open(os.path.join(train_path,"class_indices_batch_")+str(b)+".p","rb"))
        collected_class_indices.append(class_indices[0])
        count += 1

    if len(collected_batches) > 0:
        count -= 1
        print "Processed %d"%count
        m.dump(collected_batches,open(os.path.join(train_path,"bow_batch_")+str(count)+".p","wb"))
        m.dump(collected_class_indices,open(os.path.join(train_path,"class_indices_batch_")+str(count)+".p","wb"))
        batches.append(count)


    m.dump(batches,open(os.path.join(train_path,"batches.p"),"wb"))

    test_batches = m.load(open(os.path.join(test_path,"batches.p"),"rb"))
    collected_batches = []
    collected_class_indices = []
    batches = []
    count = 0
    for b in test_batches:
        if count % 100 == 0 and not count == 0:
            print "Processed %d"%count
            m.dump(collected_batches,open(os.path.join(test_path,"bow_batch_")+str(count)+".p","wb"))
            m.dump(collected_class_indices,open(os.path.join(test_path,"class_indices_batch_")+str(count)+".p","wb"))
            batches.append(count)
            collected_batches = []
            collected_class_indices = []
        data = m.load(open(os.path.join(test_path,"bow_batch_")+str(b)+".p","rb"))
        collected_batches.append(data[0])
        class_indices = m.load(open(os.path.join(test_path,"class_indices_batch_")+str(b)+".p","rb"))
        collected_class_indices.append(class_indices[0])
        count += 1

    if len(collected_batches) > 0:
        count -= 1
        print "Processed %d"%count
        m.dump(collected_batches,open(os.path.join(test_path,"bow_batch_")+str(count)+".p","wb"))
        m.dump(collected_class_indices,open(os.path.join(test_path,"class_indices_batch_")+str(count)+".p","wb"))
        batches.append(count)

    m.dump(batches,open(os.path.join(test_path,"batches.p"),"wb"))