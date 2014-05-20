import os, struct
import numpy as np
from array import array
import cPickle as pickle
import env_paths
import pylab as p


def data_processing_mnist(training = True):
    """
    Python function for importing the MNIST data set.
    """
    digits = [0,1,2,3,4,5,6,7,8,9]
    path = "input"
    if training:
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    else:
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    all_images = []
    labels = []
    for i in xrange(len(ind)):
        labels.append(lbl[ind[i]])
        tmp = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        all_images.append(np.reshape(tmp,(28*28)))



    count = 1
    batch = np.zeros((0,28*28))
    batch_lbl = np.array([])
    batches = np.array([])
    for img in all_images:
        batch = np.append(batch,np.reshape(img,((1,28*28))),axis = 0)
        batch_lbl = np.append(batch_lbl,labels[count-1])
        if count % 100 == 0:
            batches = np.append(batches,count)
            save_batch(batch/255.,batch_lbl,count,training)
            batch = np.zeros((0,28*28))
            batch_lbl = np.array([])
        count += 1

    save_batches(batches,training)

def save_batch(batch,batch_lbl,batchno,training):
    pickle.dump(batch_lbl, open(env_paths.get_class_indices_path(training,batchno), "wb"))
    pickle.dump(batch, open(env_paths.get_bow_matrix_path(training,batchno), "wb"))

def save_batches(batches,training):
    pickle.dump(batches, open(env_paths.get_batches_path(training), "wb"))

def show_image_array(images):
    for img in images:
        x = img.reshape((28,28))
        p.imshow(x,cmap=p.cm.gray)
        p.show()
