__author__ = 'larsmaaloee'
import marshal as m
import os
import shutil
from random import choice
from numpy import *
from collections import Counter
from multiprocessing import Pool

def generate_dict_for_docs(paths):
    try: return m.load(open('../input/reuters/dict_for_docs.p','rb'))
    except: print 'No serialized file...'


    data = []
    for p in paths:
        data_tmp = open(p,'rb').readlines()
        data += data_tmp

    dict_for_docs = {}
    doc_id = -1
    count = 0
    for elem in data:
        if '.I' in elem:
            count += 1
            if count % 10000 == 0:
                print 'Processed '+str(count)
            doc_id  = elem.replace('.I ','').strip()
            try: doc_id = int(doc_id)
            except: print 'Doc id not parsable'
            dict_for_docs[doc_id] = []
            continue

        if elem == '\n' or '.W' in elem:
            continue

        elem = elem.replace('\n',' ').split(' ')
        if elem == '':
            continue
        dict_for_docs[doc_id] += elem


    m.dump(dict_for_docs,open('../input/reuters/dict_for_docs.p','wb'))
    return dict_for_docs

def generate_class_dict_for_docs(path, path_topics):
    try: return m.load(open('../input/reuters/dict_classes.p','rb'))
    except: print 'No serialized file...'

    data = open(path,'rb').readlines()
    topics_lst = open(path_topics,'rb').readlines()
    topics_lst = [t.strip() for t in topics_lst]
    dict_classes = {}
    for elem in data:
        c, i, _ = elem.split(' ')
        if not c in topics_lst:
            continue
        i = int(i)
        try: dict_classes[i] += [c]
        except: dict_classes[i] = [c]

    m.dump(dict_classes,open('../input/reuters/dict_classes.p','wb'))
    return dict_classes

def generate_folders(dict_for_docs, dict_classes):
    output_path = '../input/reuters/output'
    if not os.path.exists(output_path): os.makedirs(output_path)
    for d in dict_for_docs.keys():
        doc_txt = dict_for_docs[d]
        class_ids = dict_classes[d]
        for cid in class_ids:
            class_path = os.path.join(output_path,cid)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            m.dump(doc_txt,open(class_path+'/'+str(d)+'.p','wb'))

def remove_duplicates_randomized():
    try:
        docs = m.load(open('../input/reuters/docs.p','rb'))
        docs_paths = m.load(open('../input/reuters/docs_paths.p','rb'))
    except:
        print 'No serialized file...'
        path = "../input/reuters/output"
        dirs = os.listdir(path)
        docs = []
        docs_paths = []
        for d in dirs:
            dirpath = os.path.join(path,d)
            files = os.listdir(dirpath)
            docs += files
            files = [os.path.join(dirpath,f) for f in files]
            docs_paths += files

        m.dump(docs,open("../input/reuters/docs.p","rb"))
        m.dump(docs_paths,open("../input/reuters/docs_paths.p","rb"))

    d = Counter(docs)
    no_keys = len(d.keys())
    counter = 0
    docs_arr = array(docs)
    for doc in d.keys():
        counter += 1
        if counter % 100 == 0:
            print "processed %d of %d"%(counter,no_keys)
        if d[doc] > 1:
            indices = list(where(docs_arr == doc)[0])
            # randomly remove the duplicates
            for i in range(len(indices)-1):
                idx = choice(indices)
                indices.remove(idx)
                os.remove(docs_paths[idx])

def count_docs():
    path = "../input/reuters/output"
    #path = "/Volumes/Macintosh HD/Master/data/Reuters/docs/all docs"
    docs = 0
    for folder in os.listdir(path):
        if folder.startswith("."): continue
        docs += len(os.listdir(os.path.join(path,folder)))

    print docs

def rename_folders(path,class_hier_path):
    classes = os.listdir(path)

    class_hier_file_lst = open(class_hier_path,'rb').readlines()
    class_hier_dict = {}
    for elem in class_hier_file_lst:
        splt = elem.split('  ')
        splt = [s for s in splt if not s == '']
        _, elem1, elem2 = splt
        elem1 = elem1.replace('child: ','').replace(' ','')
        elem2 = elem2.replace('child-description: ','')
        class_hier_dict[elem1] = elem2.strip()

    for c in classes:
        c_path = os.path.join(path,c)
        try: os.rename(c_path,os.path.join(path,class_hier_dict[c].replace('/',' ')))
        except: continue

def generate_white_lst(path):
    output = []
    with open(path,'rb') as f:
        for line in f:
            term, _, _ = line.split(' ')
            output.append(term)

    f = open('../input/reuters/acceptance_lst_stemmed.txt','wb')
    for item in output:
        f.write("%s\n" % item)


def generate_train_test_sets(path):
    train_path = '../input/train'
    test_path = '../input/test'
    os.makedirs(train_path)
    os.makedirs(test_path)

    class_dirs = os.listdir(path)
    for d in class_dirs:
        if d.startswith("."): continue
        docs = os.listdir(os.path.join(path,d))
        train_docs = docs[:len(docs)/2]
        test_docs = docs[len(docs)/2:]
        for traindoc in train_docs:
            if not os.path.exists(os.path.join(train_path,d)): os.makedirs(os.path.join(train_path,d))
            shutil.copyfile(os.path.join(path,d,traindoc),os.path.join(train_path,d,traindoc))

        for testdoc in test_docs:
            if not os.path.exists(os.path.join(test_path,d)): os.makedirs(os.path.join(test_path,d))
            shutil.copyfile(os.path.join(path,d,testdoc),os.path.join(test_path,d,testdoc))

if __name__ == '__main__':

    '''
    print 'Generating doc tokens dict'
    paths = ['../input/reuters/lyrl2004_tokens_test_pt0.dat','../input/reuters/lyrl2004_tokens_test_pt1.dat','../input/reuters/lyrl2004_tokens_test_pt2.dat','../input/reuters/lyrl2004_tokens_test_pt3.dat','../input/reuters/lyrl2004_tokens_train.dat']
    d = generate_dict_for_docs(paths)
    print 'Generating classes dict'
    path = '../input/reuters/rcv1-v2.topics.qrels'
    path_topics = '../input/reuters/rcv1.topics.txt'
    c = generate_class_dict_for_docs(path,path_topics)
    print 'Generating folder structure'
    generate_folders(d,c)
    print 'Removing duplicates'
    remove_duplicates_randomized()
    '''



    #rename_folders('../input/reuters/output','../input/reuters/rcv1.topics.hier.orig.txt')

    #generate_white_lst('../input/reuters/stem.termid.idf.map.txt')


    generate_train_test_sets('../input/reuters/output')

    #count_docs()