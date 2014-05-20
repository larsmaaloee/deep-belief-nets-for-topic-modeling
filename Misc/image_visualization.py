__author__ = 'larsmaaloee'

import DBN.dbn as dbn
from numpy import *
from random import choice
import serialization as s
import env_paths
from DataPreparation import data_processing
import matplotlib.pyplot as plt

def compare_real_data_to_reconstructed_data_random():
    weights = s.load(open(env_paths.get_dbn_weight_path(),"rb"))
    batches = s.load(open(env_paths.get_batches_path(train=False),"rb"))
    batch = choice(batches) # make sure to pick batch at random
    data = data_processing.get_bag_of_words_matrix(batch,training = False)
    # choose 10 data points at random
    data_points = []
    indices = random.randint(0,len(data),10)
    for idx in indices:
        data_points.append(data[idx])

    output_data_points = []
    for d in data_points:
        d = append(d,1.)
        out = generate_output_data(d,weights)
        output_data_points.append(out)

    visualise_data_points(data_points,output_data_points)

def compare_real_data_to_reconstructed_data():
    weights = s.load(open(env_paths.get_dbn_weight_path(),"rb"))
    batches = s.load(open(env_paths.get_batches_path(train=False),"rb"))
    class_indices = s.load(open(env_paths.get_class_indices_path(False,batches[0]).replace(".0",""),"rb"))
    batch = batches[0]
    data = data_processing.get_bag_of_words_matrix(batch,training = False)


    dict = {}
    for i in range(len(class_indices)):
        idx = class_indices[i]
        if idx in dict.keys(): continue
        dict[idx] = data[i]
        if len(dict) >= 10:
            break

    print dict.keys()

    data_points = dict.values()

    output_data_points = []
    for d in data_points:
        d = append(d,1.)
        out = generate_output_data(d,weights)
        output_data_points.append(out)

    visualise_data_points(data_points,output_data_points)


def generate_output_data(x, weight_matrices_added_biases):
    """
    Run through the deep autoencoder and compute the output.

    @param x: The matrix.
    @param weight_matrices_added_biases: The weight matrices added biases.
    """
    z_values = []
    for i in range(len(weight_matrices_added_biases)-1):
        if i == 0:
            z = dbn.sigmoid(dot(x,weight_matrices_added_biases[i]))
        elif i == (len(weight_matrices_added_biases)/2)-1:
            z = dot(z_values[i-1],weight_matrices_added_biases[i])
        else:
            z = dbn.sigmoid(dot(z_values[i-1],weight_matrices_added_biases[i]))

        z = append(z,1.)
        z_values.append(z)

    xout = dbn.sigmoid(dot(z_values[-1],weight_matrices_added_biases[-1]))
    return xout

def visualise_data_points(real_data, reconstructed_data):
    f,axes = plt.subplots(nrows=2, ncols=10,sharex=True,sharey=True)
    dimensions = int(sqrt(len(real_data[0])))
    for i in range(2):
        for j in range(len(real_data)):
            if i == 0:
                real_image = real_data[j].reshape((dimensions,dimensions))
                axes[i][j].imshow(real_image,cmap=plt.cm.gray)
            elif i == 1:
                recon_image = reconstructed_data[j].reshape((dimensions,dimensions))
                axes[i][j].imshow(recon_image,cmap=plt.cm.gray)
            axes[i][j].axes.get_xaxis().set_ticks([])
            axes[i][j].axes.get_yaxis().set_ticks([])

    plt.show()