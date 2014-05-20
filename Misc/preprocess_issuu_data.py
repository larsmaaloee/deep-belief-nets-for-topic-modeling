import os
from multiprocessing import Process
import json
import shutil

def preprocess_data(path):
    '''
    Filter the Issuu data to be as clean as
    possible. We do not want documents spanning
    over several categories.
    '''
    document_groups = os.listdir(path)

    evaluations = []
    for d in document_groups:
        if not d.startswith('.'):
            p = os.path.join(path,d)
            evaluations.append(__process_document_group__(p))

    jobs = [Process(j) for j in evaluations]
    for job in jobs:
        job.daemon = True
        job.start()

    for job in jobs: job.join()


def __process_document_group__(path):
    category_name = path.split('/')[-1]
    new_path = os.path.join(path.split('/')[0],"_"+category_name+"_new")
    os.mkdir(new_path)

    text_path = os.path.join(path,'Text')
    meta_path = os.path.join(path,'Meta')

    doc_paths = os.listdir(text_path)
    doc_names = []
    for d in doc_paths: doc_names.append(d.split('.')[0])

    # Split docnames into 8 batches so that we can start 8 processes
    batch_size = len(doc_names)/8
    batches = []
    for i in range(0,8,batch_size):
        batches.append(doc_names[i:i+batch_size])

    if not len(doc_names) % 8 == 0:
        batches.append(doc_names[-1-(len(doc_names)%8):])


    evaluations = []
    for b in batches:
        evaluations.append(__handle_batch__(b,text_path,meta_path,category_name,new_path))

    jobs = [Process(j) for j in evaluations]
    for job in jobs:
        job.daemon = True
        job.start()

    for job in jobs: job.join()


def __handle_batch__(batch,text_path,meta_path,category_name,new_path):
    for d in batch:
        if d == "": continue
        meta_data = open(os.path.join(meta_path,d+'.json'))
        meta_json = json.load(meta_data)
        if len(meta_json["Label"]) == 1 and meta_json["Label"][0] == category_name:
            shutil.copy(os.path.join(text_path,d+".txt"),os.path.join(new_path,d+".txt"))