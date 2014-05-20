'''
Created on Sep 26, 2013

@author: larsmaaloee
'''

import getopt
from numpy import *
from Testing import dbn_testing
from DataPreparation import data_processing
from DBN import dbn
import sys
import os

def usage(v,h,o,e,b):
    print 'DBN for Topic Modeling'
    print '$Id: v 2.0b 2014/14/04$'
    print 'usage  : cmd_tool.py aut/man/test [options]'
    print 'options: -V visible number of visible variables default: ',v
    print '         -H hidden number of hidden variables default: ',h
    print '         -O output number of output variables default: ',o
    print '         -E epochs number of epochs default: ',e
    print '         -B should the DBN output binary or real numbered values (true/false): ',b
    sys.exit (0)

def handle_sys_args(sys_argv):
    #test = ["man","-V 2000", "-H 500,250", "-O 2","-E 1","-T 0.7"]
    #sys.argv[1:] = test
    shortoptions = "V:H:O:E:B"
    longoptions = ["visible=","hidden=","output=","epochs=","binary="]

    visible = 2000
    hiddentxt = "500,500"
    hidden = [500,500]
    output = 128
    epochs = 50
    train = 0.7
    path = 'input'

    if len(sys_argv)<2:
        usage(visible,hiddentxt,output,epochs,train)
    try:
        opts,_ = getopt.getopt(sys.argv[2:], shortoptions, longoptions)
    except:
        usage(visible,hiddentxt,output,epochs,train)


    # parse arguments
    for o, a in opts:
        if o in ('-V', '--visible'):
            visible = int(a)
        elif o in ('-H', '--hidden'):
            hidden = []
            hiddens = a.split(',')
            for h in hiddens:
                hidden.append(int(h))
        elif o in ('-O', '--output'):
            output = int(a)
        elif o in ('-E', '--epochs'):
            epochs = int(a)
        elif o in ('-B', '--binary'):
            bin = str(a)
        else:
            assert False, "unknown option"

    if 'y' in raw_input("Datapreparation? [y/n]"):
        check_path(path)
        training_path = raw_input("Enter training path as a relative path (i.e. input/train):")
        test_path = raw_input("Enter testing path as a relative path (i.e. input/test):")
        stem = raw_input("Stemming the documents is needed for the data processing to complete. Stem documents? [y/n]")
        run_data_processing(visible,training_path,test_path,stem = True if stem == 'y' else False)

    if 'aut' in sys_argv[1]:
        deepbelief = dbn.DBN(visible, data_processing.get_batch_list(), hidden, output, epochs,binary_output=bin)
        deepbelief.run_dbn()

    elif 'man' in sys_argv[1]:
        deepbelief = dbn.DBN(visible, data_processing.get_batch_list(), hidden, output, epochs,binary_output=bin)
        if 'y' in raw_input("Pre-training? [y/n]"):
            deepbelief.run_pretraining()
        if 'y' in raw_input("Fine-tuning? [y/n]"):
            deepbelief.run_finetuning(load_from_serialization=True)

def check_path(p):
    try:
        if(len(os.listdir(p))==0):
            print "The '/input' directory is empty."
            sys.exit(0)
    except:
        print "There exist no '/input' directory."
        sys.exit(0)

def run_data_processing(attributes,train_path,test_path,stem = True):
    paths = os.listdir(train_path)
    train_paths= []
    for p in paths:
        if p.startswith('.'):
            continue
        train_paths.append(os.path.join(train_path, p))

    paths = os.listdir(test_path)
    test_paths= []
    for p in paths:
        if p.startswith('.'):
            continue
        test_paths.append(os.path.join(test_path, p))

    if stem:
        # Stem documents
        data_processing.stem_docs(train_paths)
        data_processing.stem_docs(test_paths)

    dat_proc_train = data_processing.DataProcessing(train_paths,words_count=attributes,batchsize=10,trainingset_size=1.0)
    dat_proc_train.generate_bows()
    dat_proc_test = data_processing.DataProcessing(test_paths,trainingset_size=0.0, batchsize=10,trainingset_attributes=data_processing.get_attributes())
    dat_proc_test.generate_bows()

if __name__ == '__main__':
    handle_sys_args(sys.argv)