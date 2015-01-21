'''
Created on Sep 23, 2013

@author: larsmaaloee
'''

import os
from os.path import join
import time
import datetime
import shutil


# Data processing

def get_data_path(train):
    return check_dir(join(output_path(),"train")) if train else check_dir(join(output_path(),"test"))

def get_bow_path(train):
    return check_dir(join(get_data_path(train),"BOWs"))

def get_doc_load_path(train):
    return check_dir(join(get_data_path(train),"stemmed_docs_lsts"))

def get_class_names_path(train):
    return join(get_bow_path(train),"class_names.p")

def get_batches_path(train):
    return join(get_bow_path(train),"batches.p")

def get_attributes_path(train):
    return join(get_bow_path(train),"attributes.p")

def get_doc_list_path(train,batch):
    return join(get_doc_load_path(train),"docs_list"+"_batch_"+str(batch)+".p")

def get_doc_names_path(train,batch):
    return join(get_bow_path(train),"docs_names_batch_"+str(batch)+".p")

def get_class_indices_path(train,batch):
    return join(get_bow_path(train),"class_indices_batch_"+str(batch)+".p")

def get_bow_matrix_path(train,batch):
    return join(get_bow_path(train),"bow_batch_"+str(batch)+".p")






# RBM
def get_rbm_data_path():
    return check_dir(join(output_path(),"rbm"))

def get_rbm_output_path(units,batch,layer_index):
    return join(get_rbm_data_path(),"output_"+str(units)+"_batch_"+str(batch)+"_layer_"+str(layer_index)+".p")

def get_rbm_plotting_output(epoch,layer):
    return join(get_rbm_data_path(),"layer_"+str(layer)+"_epoch_"+str(epoch)+"_out")

def get_rbm_plotting_input(epoch,layer):
    return join(get_rbm_data_path(),"layer_"+str(layer)+"_epoch_"+str(epoch)+"_in")

def get_rbm_weights_path():
    return join(get_rbm_data_path(),"weight_matrices.p")

def get_rbm_hidden_biases_path():
    return join(get_rbm_data_path(),"hidden_biases.p")

def get_rbm_visible_biases_path():
    return join(get_rbm_data_path(),"visible_biases.p")

def get_rbm_output_txt_path():
    return join(get_rbm_data_path(),"output.txt")







# DBN
def get_dbn_data_path():
    return check_dir(join(output_path(),"dbn"))

def get_dbn_weight_path():
    return join(get_dbn_data_path(),"weight_matrices.p")

def get_dbn_training_error_path():
    return join(get_dbn_data_path(),"train_error.p")

def get_dbn_test_error_path():
    return join(get_dbn_data_path(),"test_error.p")

def get_dbn_output_txt_path():
    return join(get_dbn_data_path(),"output.txt")

def get_dbn_large_batch_data_path(batch):
    return join(check_dir(join(get_dbn_data_path(),'bag_of_words')),str(batch)+'.p')

def get_dbn_batches_lst_path():
    return join(check_dir(join(get_dbn_data_path(),'bag_of_words')),'batches.p')


def input_path():
    return check_dir("input")

def output_path():
    return check_dir("output")


def archive_outputs():
    # Make dir for archiving
    t = time.time()
    arch_dir_n = "_arch_"+datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H%M%S')
    arch_dir_fp = os.path.join(output_path(),arch_dir_n)
    os.makedirs(arch_dir_fp)

    # Move output files to archive
    for f in os.listdir(output_path()):
        if f.startswith("_") or f.startswith("."): continue
        shutil.move(os.path.join(output_path(),f),arch_dir_fp)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path