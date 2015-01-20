__author__ = 'larsmaaloee'

import pylab as Plot
import matplotlib.pyplot as plt
from DBN.dbn import generate_input_data_list, generate_output_for_test_data, generate_output_for_train_data
from pca import pca_2d, pca_3d, pca_2d_for_2_components, pca_3d_movie
from DataPreparation.data_processing import get_all_class_indices, get_all_class_names, \
    get_class_names_for_class_indices
import env_paths as ep
import serialization as s
from numpy import *


class Visualise:
    def __init__(self, testing=True, classes_to_visualise=None, image_data=False, binary_output=False):
        """
        @param testing: Should be True if test data is to be plottet. Otherwise False.
        @param classes_to_visualise: A list containing the classes to visualise
        @param image_data: If the visualization should be done on image data.
        @param binary_output: If the output of the DBN must be binary.
        """

        if not check_for_data:
            print 'No DBN data or testing data.'
            return
        self.path = "output"
        self.output = []
        self.testing = testing
        self.input_data = []
        self.output_data = []
        self.classes_to_visualise = classes_to_visualise
        self.image_data = image_data
        self.binary_output = binary_output

    def __generate_input_data(self):
        """
        Generate the input data for the DBN so that it can be visualized.
        """
        if not len(self.input_data) == 0:
            return

        try:
            self.input_data = s.load(open('output/input_data.p', 'rb'))
            self.class_indices = s.load(open('output/class_indices.p', 'rb'))
            if not self.classes_to_visualise == None:
                self.__filter_input_data(self.classes_to_visualise)
        except:
            self.input_data = generate_input_data_list(training=False) if self.testing else generate_input_data_list()
            self.class_indices = get_all_class_indices(training=False) if self.testing else get_all_class_indices()
            if not self.classes_to_visualise == None:
                self.__filter_input_data(self.classes_to_visualise)
            s.dump([input.tolist() for input in self.input_data], open('output/input_data.p', 'wb'))
            s.dump(self.class_indices, open('output/class_indices.p', 'wb'))

        self.legend = get_class_names_for_class_indices(list(set(sorted(self.class_indices))))


    def __generate_output_data(self):
        """
        Generate the output data of the DBN so that it can be visualised.
        """
        if not len(self.output_data) == 0:
            return
        try:
            self.output_data = s.load(open('output/output_data.p', 'rb'))
            self.class_indices = s.load(open('output/class_indices.p', 'rb'))
            if not self.classes_to_visualise == None:
                self.__filter_output_data(self.classes_to_visualise)
        except:
            self.output_data = generate_output_for_test_data(image_data=self.image_data,
                                                             binary_output=self.binary_output) if self.testing else generate_output_for_train_data(
                image_data=self.image_data, binary_output=self.binary_output)
            self.class_indices = get_all_class_indices(training=False) if self.testing else get_all_class_indices()
            if not self.classes_to_visualise == None:
                self.__filter_output_data(self.classes_to_visualise)
            s.dump([out.tolist() for out in self.output_data], open('output/output_data.p', 'wb'))
            s.dump(self.class_indices, open('output/class_indices.p', 'wb'))

        self.legend = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                       "9"]  # get_class_names_for_class_indices(list(set(sorted(self.class_indices))))

    def __filter_output_data(self, classes_to_visualise):
        """
        Filter the output or input data corresponding to the classes to visualise.

        @param classes_to_visualise: A list containing names for the classes to visualise.
        """
        class_names = get_all_class_names()
        class_indices_to_visualise = []
        for i in range(len(class_names)):
            if class_names[i] in classes_to_visualise:
                class_indices_to_visualise.append(i)

        if not len(class_indices_to_visualise) == len(classes_to_visualise):
            print 'Not all classes to visualise were correct.'
            return

        tmp_output_data = []
        tmp_class_indices = []
        for i in range(len(self.output_data)):
            out = self.output_data[i]
            idx = self.class_indices[i]
            if idx in class_indices_to_visualise:
                tmp_output_data.append(out)
                tmp_class_indices.append(idx)

        self.output_data = tmp_output_data
        self.class_indices = tmp_class_indices

    def __filter_input_data(self, classes_to_visualise):
        """
        Filter the output or input data corresponding to the classes to visualise.

        @param classes_to_visualise: A list containing names for the classes to visualise.
        """
        class_names = get_all_class_names()
        class_indices_to_visualise = []
        for i in range(len(class_names)):
            if class_names[i] in classes_to_visualise:
                class_indices_to_visualise.append(i)

        if not len(class_indices_to_visualise) == len(classes_to_visualise):
            print 'Not all class names were correct.'
            return

        tmp_output_data = []
        tmp_class_indices = []
        for i in range(len(self.input_data)):
            out = self.input_data[i]
            idx = self.class_indices[i]
            if idx in class_indices_to_visualise:
                tmp_output_data.append(out)
                tmp_class_indices.append(idx)

        self.input_data = tmp_output_data
        self.class_indices = tmp_class_indices


    def visualise_2d_data(self):
        """
        In case the number of output units of the DBN is 2, this method will visualise the output of the DBN on a
        2D plot.
        """
        self.__generate_output_data()
        if len(self.output_data[0]) != 2:  # The output dimensions must be 2
            return
        f = Plot.figure()
        f.hold()
        plt.title('2D data')
        for c in sorted(set(self.class_indices)):
            class_mask = mat(self.class_indices).T.A.ravel() == c
            plt.plot(array(self.output_data)[class_mask, 0], array(self.output_data)[class_mask, 1], 'o')
        plt.legend(self.legend)
        plt.show()
        plt.savefig(self.path + '/2dplotlow.png', dpi=200)

    def visualise_data_pca_2d(self, input_data=False, number_of_components=9):
        """
        Visualise the input data or the output data of the DBN on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.

        @param input_data: False if the output data of the DBN should be plottet. Otherwise True.
        @param number_of_components: The number of principal components for the PCA plot.
        """

        if input_data:
            self.__generate_input_data()
            pca_2d(array(self.input_data), self.class_indices, self.path, 'high_dimension_data', number_of_components,
                   self.legend)
        else:
            self.__generate_output_data()
            pca_2d(array(self.output_data), self.class_indices, self.path, 'low_dimension_data', number_of_components,
                   self.legend)

    def visualise_data_pca_2d_two_components(self, component1, componen2, input_data=False):
        """
        Visualise the input data or the output data of the DBN on a 2D PCA plot. Specify two components, which will be
        the plotted.

        @param input_data: False if the output data of the DBN should be plottet. Otherwise True.
        @param component1: Principal component 1.
        @param componen2: Principal component 2.
        """
        if input_data:
            self.__generate_input_data()
            pca_2d_for_2_components(array(self.input_data), component1, componen2, self.class_indices, self.path,
                                    'high_dimension_data', self.legend)
        else:
            self.__generate_output_data()
            pca_2d_for_2_components(array(self.output_data), component1, componen2, self.class_indices, self.path,
                                    'low_dimension_data', self.legend)

    def visualise_data_pca_3d(self, component1, component2, component3, input_data=False):
        """
        Visualise the input data or the output data of the DBN on a 3D PCA plot for principal components 1 and 2.

        Parameters
        ----------
        input_data: False if the output data of the DBN should be plottet. Otherwise True.
        """
        if input_data:
            self.__generate_input_data()
            pca_3d(array(self.input_data), component1, component2, component3, self.class_indices, self.path,
                   'high_dimension_data', self.legend)
        else:
            self.__generate_output_data()
            pca_3d(array(self.output_data), component1, component2, component3, self.class_indices, self.path,
                   'low_dimension_data', self.legend)

    def visualise_data_pca_3d_movie(self, component1, component2, component3, input_data=False):
        """
        Visualise the input data or the output data of the DBN on a 3D PCA plot movie for principal components 1 and 2.

        Parameters
        ----------
        input_data: False if the output data of the DBN should be plottet. Otherwise True.
        """
        if input_data:
            self.__generate_input_data()
            pca_3d_movie(array(self.input_data), component1, component2, component3, self.class_indices, self.path,
                         'high_dimension_data', self.legend)
        else:
            self.__generate_output_data()
            pca_3d_movie(array(self.output_data), component1, component2, component3, self.class_indices, self.path,
                         'low_dimension_data', self.legend)


def check_for_data():
    """
    Check for DBN network data.
    """
    if not (os.path.exists(ep.get_test_data_path()) or os.path.exists(ep.get_dbn_weight_path())):
        return False
    return True
