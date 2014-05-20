'''
Created on Mar 21, 2013

@author: larsmaaloee
'''

import networkx as nx
import cPickle as p
from numpy import *
import matplotlib.pyplot as plt
from operator import itemgetter
import uuid
import os
from multiprocessing import Pool
import shutil
import marshal


def read_into_dicts():
    graphdict = {}
    labeldict = {}

    with open('../input/wikipedia_graph.txt','rb') as f:
        for line in f:
            leftright = line.split("\t")
            if not len(leftright) == 2:
                print 'Ambigouse file structure. Line: ',line
                return
            graphdict[leftright[0]] = leftright[1].replace("\n","").split(",")


    with open('../input/wikipedia_labels.txt','rb') as f:
        for line in f:
            leftright = line.split(",")
            if len(leftright)>2:
                coll = ""
                for elem in leftright[1:]:
                    coll+=elem+","
                leftright = [leftright[0],coll]

            if not len(leftright) == 2:
                print 'Ambigouse file structure. Line: ',line
                return
            labeldict[leftright[0]] = leftright[1].replace("\n","").split(",")


    dump_dicts(graphdict,labeldict)

def dump_dicts(graphdict,labeldict):
    p.dump(graphdict,open('../output/graphdict.p','wb'))
    p.dump(labeldict,open('../output/labeldict.p','wb'))

def load_dicts():
    return p.load(open('../output/graphdict.p','rb')),p.load(open('../output/labeldict.p','rb'))

def create_graph(category):
    graphdict, labeldict = load_dicts()
    G = nx.Graph()
    ego = labeldict[category]
    categories = [category]
    level = 0

    while True:
        if level == 2 or len(categories) == 0:
            break
        for cat in categories:
            node = labeldict[cat][0].decode('utf-8')
            G.add_node(node)
            try:
                subcats = graphdict[cat]
                for scat in subcats:
                    subnode = labeldict[scat][0].decode('utf-8')
                    G.add_node(subnode)
                    G.add_edge(node,subnode)
            except KeyError:
                continue
        categories = subcats
        level += 1

    # Draw graph
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos,node_size=10,node_color='white')
    nx.draw_networkx_edges(G,pos,width=0.5,alpha = 0.8, edge_color = 'black')
    nx.draw_networkx_labels(G,pos, font_size = 12, font_family = 'sans-serif')
    nx.write_gexf(G,'../output/graph.gexf')
    plt.savefig('../output/ego_graph.png')
    plt.show()


def output_docs(categories,docs_no = 24):
    paths = []
    for i in range(docs_no):
        paths.append(os.path.join('..','input','Wiki_'+str(i+1)+'.xml'))
    p = Pool(6)
    p.map_async(output_doc,[(path,categories) for path in paths])
    p.close()
    p.join()

def output_doc(args):
    path,categories = args
    with open(path,'rb') as f:
        doc = ""
        for line in f:
            if "<page>" in line or "</page>" in line:
                doc = ""

            elif "[[Category:" in line:
                for cat in categories:
                    if cat in line:
                        doc_uid = str(uuid.uuid1())+'.txt'
                        p = os.path.join('..','output','wiki_out',cat)
                        if not os.path.exists(p):
                            os.makedirs(p)
                        doc_file = open(os.path.join(p,doc_uid),'wb')
                        doc_file.write(doc)

            else:
                doc += line

def generate_train_test_sets():
    path = '../output/wiki_out'
    paths = os.listdir('../output/wiki_out')
    train_dir = '../output/wiki_out/train'
    test_dir = '../output/wiki_out/test'
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(test_dir): os.makedirs(test_dir)

    for p in paths:
        if p.startswith('.'): continue
        docs = os.listdir(os.path.join(path,p))
        train_docs = [os.path.join(path,p,d) for d in docs[-int(len(docs)*0.7):]]
        test_docs = [os.path.join(path,p,d) for d in docs[:-int(len(docs)*0.7)]]
        train_cat_path = os.path.join(train_dir,p)
        test_cat_path = os.path.join(test_dir,p)
        if not os.path.exists(train_cat_path): os.makedirs(train_cat_path)
        if not os.path.exists(test_cat_path): os.makedirs(test_cat_path)
        for d in train_docs: shutil.copyfile(d,os.path.join(train_cat_path+'/',d.split("/")[-1]))
        for d in test_docs: shutil.copyfile(d,os.path.join(test_cat_path,d.split("/")[-1]))


def write_titles_of_test_train_sets():
    print 'Retrieving docs'
    titles = {}
    test_path = "../input/test"
    categories = os.listdir(test_path)
    for c in categories:
        if c.startswith('.'): continue
        print 'Category: ',c
        catpath = os.path.join(test_path,c)
        titles_tmp = []
        for d in os.listdir(catpath):
            if not d.endswith('.txt'):
                continue
            docpath = os.path.join(catpath,d)
            with open(docpath,'rb') as f:
                for line in f:
                    if '<title>' in line:
                        title = line.replace('<title>','').replace('</title>','').replace('\n','').strip()
                        title = title.replace(',',' ')
                        titles_tmp.append((title,d,c,categories.index(c)))
        titles[c] = titles_tmp

    output_str = ""
    for k in titles.keys():
        for v in titles[k]:
            title,doc,cat,cat_no = v
            output_str += title+","+doc+","+cat+","+str(cat_no)+"\n"

    f = open('../output/titles.txt','wb')
    f.write(output_str)
    f.close()

def find_lda_docs_of_interest():
    fname = '/Volumes/Macintosh HD/Master/data/wiki/LDA_All.txt'
    titles = set([elem.split('\t,\t')[0] for elem in open('../output/titles.txt','rb').readlines()])
    #titles = [t.replace(',',' ') for t in titles]
    titles = [t.split(",")[0] for t in titles]

    count = 0
    with open(fname,'rb') as f,open('../output/output.txt','wb') as f2:
        for line in f:
            if count % 100000 == 0:
                print count
            splt_line = line.split(",")
            if splt_line[0] in titles:
                f2.write(line)
            count += 1
    print 'Done'


    wiki_titles = [line.split(",")[0] for line in open('../output/output.txt','rb').readlines()]

    print 'diff: ', list(set(titles) - set(wiki_titles))
    print 'Titles len: ',len(titles)

def generate_lda_output():
    lda_titles = []
    lda_doc_names = []
    lda_class_indices = []
    lda_coordinates = []

    f1 = open('../output/output.txt','rb').readlines()
    f2 = open('../output/titles.txt','rb').readlines()
    print 'Generate class indices for LDA'

    count = 0
    for elem1 in f1:
        count += 1
        splt = elem1.split(',')
        title = splt[0]
        coor = [float(coor) for coor in splt[2:152]]
        for elem2 in f2:
            splt2 = elem2.split(",")
            if splt2[0] == title:
                lda_titles.append(title)
                lda_doc_names.append(splt2[1])
                lda_class_indices.append(splt2[3])
                lda_coordinates.append(coor)
    print count
    p.dump(lda_titles,open('../output/lda_titles.p','wb'))
    p.dump(lda_doc_names,open('../output/lda_doc_names.p','wb'))
    p.dump(lda_class_indices,open('../output/class_indices.p','wb'))
    p.dump(lda_coordinates,open('../output/output_data.p','wb'))

def generate_bar_plot_of_doc_dist():
    path = '../output/wiki_out'
    dist = {}
    dirs = os.listdir(path)
    for d in dirs:
        if d[0] == '.':
            continue
        count = len(os.listdir('../output/wiki_out/'+d))
        dist[d] = count
    fig = plt.figure()
    x = arange(1, len(dist)+1)
    y = dist.values()
    labels = dist.keys()
    width = 1
    bar1 = plt.bar( x, y, width, color="black" )
    plt.ylabel( 'Doc counts' )
    plt.xticks(x + width/2.0, labels )

    total_count = 0
    for v in y:
        total_count += v
    print 'Total count of docs are: %d'%total_count
    plt.show()

if __name__ == '__main__':
    #write_titles_of_test_train_sets()
    #find_lda_docs_of_interest()
    #generate_lda_output()

    #read_into_dicts()
    #create_graph('563')
    #categories = ['occupations','management','marketing','commerce','globalization','business people','companies','finance','business ethics','labor','industry','business-related media','office work','family businesses','promotion and marketing communications','sales','administration','business conferences','business documents','sports business']
    #categories = ['business','occupations','finance','fashion','design','mathematics','algebra','physics','education','sports','motorsport','society','government','law','food','science']

    #output_docs(categories)
    #generate_train_test_sets()
    #generate_bar_plot_of_doc_dist()


    generate_bar_plot_of_doc_dist()