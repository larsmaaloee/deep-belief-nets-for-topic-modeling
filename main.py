__author__ = 'larsmaaloee'

import os

from Testing import dbn_testing
from Testing import visualise
from DataPreparation import data_processing
from DBN import dbn
from env_paths import archive_outputs


def example1():
    '''
    Run simulation on the 20 Newsgroups dataset "20news-bydate.tar.gz" from http://qwone.com/~jason/20Newsgroups/
    through a network structure 2000-500-500-128 (binary outputs).
    '''

    ### Archiving output files ###
    # Archive output files so that the new simulation for example 1 will not use the data already present.
    archive_outputs()

    ### DATA PREPARATION ###

    # Define training and test set paths.
    train_path = os.path.join('input','20news-bydate','20news-bydate-train')
    test_path = os.path.join('input','20news-bydate','20news-bydate-test')

    # Generate list of all the subfolders in the training path
    paths = os.listdir(train_path)
    train_paths = []
    for p in paths:
        if p.startswith('.'):  # check for hidden files
            continue
        train_paths.append(os.path.join(train_path, p))
    print train_paths

    # Generate list of all the subfolders in the test path
    paths = os.listdir(test_path)
    test_paths = []
    for p in paths:
        if p.startswith('.'):  # check for hidden files
            continue
        test_paths.append(os.path.join(test_path, p))
    print test_paths

    # Stem documents and compute a .p file (serlialized file).
    data_processing.stem_docs(train_paths)
    data_processing.stem_docs(test_paths)

    # Generate bag of word matrix for training set.
    dat_proc_train = data_processing.DataProcessing(train_paths, words_count=2000, trainingset_size=1.0)
    dat_proc_train.generate_bows()
    # Generate bag of word matrix for test set.
    dat_proc_test = data_processing.DataProcessing(test_paths, trainingset_size=0.0,
                                                   trainingset_attributes=data_processing.get_attributes())
    dat_proc_test.generate_bows()

    ### DBN TRAINING ###

    # Generate network 2000-500-500-128 (binary outputs), training 50 epochs.
    deepbelief = dbn.DBN(2000, data_processing.get_batch_list(), [500, 500], 128, 5, binary_output=True)
    # Pretrain with a replicated softmax model at the bottom and restricted boltzmann machines in the remaining layers.
    deepbelief.run_pretraining()
    # Construct deep autoencoder and finetune using backpropagation with conjugate gradient as optimization.
    deepbelief.run_finetuning(load_from_serialization=True)

    ### EVALUATION ###

    # Evaluate on the test set and output as binary output units.
    eval = dbn_testing.DBNTesting(testing=True, binary_output=True)
    # Evaluate the output space on the 1,3,7,15 nearest neighbors.
    eval.generate_accuracy_measurement_parallel([1, 3, 7, 15])

    ### VISUALISATION ###

    # Initialise visualization. Only plot 6 categories so that the plot will not get too cluttered.
    v = visualise.Visualise(testing=True, classes_to_visualise=["rec.sport.hockey", "comp.graphics", "sci.crypt",
                                                                "soc.religion.christian", "talk.politics.mideast",
                                                                "talk.politics.guns"])
    # Visualise the output data with 4 principal components.
    v.visualise_data_pca_2d(input_data=False, number_of_components=4)
    # Visualise the output data with 2 principal components.
    v.visualise_data_pca_2d_two_components(1, 2, input_data=False)
    # Visualise the output data in 3d with 3 principal components.
    v.visualise_data_pca_3d(1, 2, 3, input_data=False)
    # Visualise the output in a 3D movie.
    v.visualise_data_pca_3d_movie(1, 2, 3, input_data=False)


def example2():
    '''
    Run simulation on the 20 Newsgroups dataset "20news-18828.tar.gz" from http://qwone.com/~jason/20Newsgroups/
    through a network structure 2000-500-250-125-10 (real valued outputs).
    '''

    ### Archiving output files ###
    # Archive output files so that the new simulation for example 1 will not use the data already present.
    archive_outputs()

    ### DATA PREPARATION ###

    # Define training and test set paths.
    datapath = os.path.join('input','20news-18828')

    # Generate list of all the subfolders in the data path
    paths = os.listdir(datapath)
    datapaths = []
    for p in paths:
        if p.startswith('.'):  # check for hidden files
            continue
        datapaths.append(os.path.join(datapath, p))
    print datapaths

    # Stem documents and compute a .p file (serlialized file).
    data_processing.stem_docs(datapaths)

    # Generate bag of word matrix for training set which is 0.7 (70%) of the data in the data paths.
    dat_proc_train = data_processing.DataProcessing(datapaths, words_count=2000, trainingset_size=1.0)
    dat_proc_train.generate_bows()
    # Generate bag of word matrix for test set which is 0.3 (30%) of the data in the data paths.
    dat_proc_test = data_processing.DataProcessing(datapaths, trainingset_size=0.0,
                                                   trainingset_attributes=data_processing.get_attributes())
    dat_proc_test.generate_bows()

    ### DBN TRAINING ###

    # Generate network 2000-500-250-125-10 (real valued outputs), training 50 epochs.
    deepbelief = dbn.DBN(2000, data_processing.get_batch_list(), [500, 250, 125], 10, 50, binary_output=False)
    # Pretrain with a replicated softmax model at the bottom and restricted boltzmann machines in the remaining layers.
    deepbelief.run_pretraining()
    # Construct deep autoencoder and finetune using backpropagation with conjugate gradient as optimization.
    deepbelief.run_finetuning(load_from_serialization=True)

    ### EVALUATION ###

    # Evaluate on the test set and output as real output units.
    eval = dbn_testing.DBNTesting(testing=True, binary_output=False)
    # Evaluate the output space on the 1,3,7,15 nearest neighbors.
    eval.generate_accuracy_measurement_parallel([1, 3, 7, 15])

    ### VISUALISATION ###

    # Initialise visualization. Only plot 6 categories so that the plot will not get too cluttered.
    v = visualise.Visualise(testing=True, classes_to_visualise=["rec.sport.hockey", "comp.graphics", "sci.crypt",
                                                                "soc.religion.christian", "talk.politics.mideast",
                                                                "talk.politics.guns"])
    # Visualise the output data with 4 principal components.
    v.visualise_data_pca_2d(input_data=False, number_of_components=4)
    # Visualise the output data with 2 principal components.
    v.visualise_data_pca_2d_two_components(1, 2, input_data=False)
    # Visualise the output data in 3d with 3 principal components.
    v.visualise_data_pca_3d(1, 2, 3, input_data=False)
    # Visualise the output in a 3D movie with 3 principal components.
    v.visualise_data_pca_3d_movie(1, 2, 3, input_data=False)


def example3():
    '''
    In the output folder ./output you'll find "20news-19997.tar.gz" from http://qwone.com/~jason/20Newsgroups/
    processed so that you can run evaluation on the data directly. The data has been trained through a network
    of 2000-500-250-125-10 (real valued output). Unzip the compressed chunks by running the output/_unzip.sh shell
    script.
    '''
    ### EVALUATION ###

    # Evaluate on the test set and output as real output units.
    eval = dbn_testing.DBNTesting(testing=True, binary_output=False)
    # Evaluate the output space on the 1,3,7,15 nearest neighbors.
    eval.generate_accuracy_measurement_parallel([1, 3, 7, 15])

    ### VISUALISATION ###

    # Initialise visualization. Only plot 6 categories so that the plot will not get too cluttered.
    v = visualise.Visualise(testing=True, classes_to_visualise=["rec.sport.hockey", "comp.graphics", "sci.crypt",
                                                                "soc.religion.christian", "talk.politics.mideast",
                                                                "talk.politics.guns"])
    # Visualise the output data with 4 principal components.
    v.visualise_data_pca_2d(input_data=False, number_of_components=4)
    # Visualise the output data with 2 principal components.
    v.visualise_data_pca_2d_two_components(1, 2, input_data=False)
    # Visualise the output data in 3d with 3 principal components.
    v.visualise_data_pca_3d(1, 2, 3, input_data=False)


def run_examples():
    '''
    In order to run example 1 and example 2, please download from http://qwone.com/~jason/20Newsgroups/:
    Example1: "20news-bydate.tar.gz"
        (save training data to ./input/20news-bydate-small/train and test data to ./input/20news-bydate-small/train.)
    Example2: "20news-18828.tar.gz"
        (save all data to ./input/20_newsgroups)
    Example3: Runs out of the box on the output data given in ./output folder.
    '''
    #example1()
    #example2()
    example3()


if __name__ == '__main__':
    run_examples()
