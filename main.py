__author__ = 'larsmaaloee'


import os
from Testing import dbn_testing
from Testing import visualise
from DataPreparation import data_processing
from DBN import dbn

def run_vis():
    # Initialise visualization. Only plot 6 categories so that the plot will not get cluttered.
    v = visualise.Visualise(testing=True,classes_to_visualise=["rec.sport.hockey","comp.graphics","sci.crypt","soc.religion.christian","talk.politics.mideast","talk.politics.guns"])
    # Visualise the output data with 4 principal components
    v.visualise_data_pca_2d(input_data=False,number_of_components=4)
    # Visualise the output data with 2 principal components
    v.visualise_data_pca_2d_two_components(1,2,input_data=False)
    # Visualise the output in a 3D movie
    v.visualise_data_pca_3d_movie(1,2,3,input_data=False)

def run_simulation(train_path, test_path, epochs = 50, attributes = 2000, evaluation_points = [1,3,7,15,31,63],binary_output = True):
    # Define training and test set paths.
    paths = os.listdir(train_path)
    train_paths= []
    for p in paths:
        if p.startswith('.'):
            continue
        train_paths.append(os.path.join(train_path,p))

    print train_paths

    paths = os.listdir(test_path)
    test_paths= []
    for p in paths:
        if p.startswith('.'):
            continue
        test_paths.append(os.path.join(test_path,p))
    print test_paths

    # Stem documents
    #data_processing.stem_docs(train_paths)
    #data_processing.stem_docs(test_paths)

    # Generate bag of word matrice
    dat_proc_train = data_processing.DataProcessing(train_paths,words_count=attributes,trainingset_size=1.0,acceptance_lst_path="input/acceptance_lst_stemmed.txt")
    #dat_proc_train.generate_bows()
    dat_proc_test = data_processing.DataProcessing(test_paths,trainingset_size=0.0, trainingset_attributes=data_processing.get_attributes())
    dat_proc_test.generate_bows()

    # Train network
    deepbelief = dbn.DBN(attributes, data_processing.get_batch_list(), [500], 500, epochs, binary_output=binary_output)
    deepbelief.run_pretraining()
    deepbelief.run_finetuning(load_from_serialization=True)

    # Evaluate network
    test = dbn_testing.DBNTesting(testing = True,binary_output=False)
    test.generate_accuracy_measurement_parallel(evaluation_points)


if __name__ == '__main__':
    run_simulation('input/20news-bydate/20news-bydate-train','input/20news-bydate/20news-bydate-test',epochs = 50,attributes=2000,evaluation_points=[1,3,7,15,31,63],binary_output=True)
    run_vis()