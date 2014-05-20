'''
Created on Sep 11, 2013

@author: larsmaaloee
'''

import pylab as Plot

from DataPreparation import data_processing
from Misc.RSM import tsne, rsm
from Testing.pca import *


class Visualise():
    def __init__(self,trainingdata = False):
        self.batches = data_processing.get_batch_list(trainingdata)        
        
        # Run data through neural network
        self.lower_dimension_data = [] # Output data from the dbn
        self.higher_dimensional_data = [] # Input data to the dbn
        
        
        self.path = 'output'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        weights = rsm.get_weights()
        visible_biases = rsm.get_visible_biases()
        hidden_biases = rsm.get_hidden_biases()
        
        # Generate class indices and class names
        if trainingdata:
            path = 'pickle/train/bag_of_words'
        else:
            path = 'pickle/test/bag_of_words'
        
        self.class_indices = self.__generate_class_indices__(path, self.batches) # Class indices for all documents
        
        # Run through batches and generate high and low dimensional data lists
        for batch in range(len(self.batches)):
            print 'Batch ',batch + 1, ' of ',len(self.batches)
            d = data_processing.get_bag_of_words_matrix(self.batches[batch],trainingdata)
            self.higher_dimensional_data += list(d)
            self.lower_dimension_data += list((rsm.generate_output_data(d, weights,visible_biases,hidden_biases)))

   
           
    
    def visualise_2d_data(self):
        
        # The output dimensions must to be 2
        if len(self.lower_dimension_data[0]) != 2:
            return
        
        f = figure()
        f.hold()
        title('2D data')
        
        for c in sorted(set(self.class_indices)):
            class_mask = np.mat(self.class_indices).T.A.ravel()==c
            plot(array(self.lower_dimension_data)[class_mask,0], array(self.lower_dimension_data)[class_mask,1], 'o')
        
        legend(sorted(set(self.class_indices)))
        savefig(self.path+'/2dplotlow.png',dpi=200)
        show()    
        
    def visualise_data_pca_2d(self,number_of_components = 9):
        # PCA
        pca_2d(array(self.lower_dimension_data),self.class_indices,self.path,'low_dimension_data',number_of_components)
        pca_2d(array(self.higher_dimensional_data),self.class_indices,self.path,'high_dimension_data',number_of_components)
    
    def visualise_data_pca_3d(self):
        # PCA
        pca_3d(array(self.lower_dimension_data),self.class_indices,self.path,'low_dimension_data')
        pca_3d(array(self.higher_dimensional_data),self.class_indices,self.path,'high_dimension_data')
        
    
    def visualise_data_sne(self,path):
        # Convert to numpy array
        X = array(self.lower_dimension_data,dtype = float64)
        # Stochastic neighbor embedding
        Y = tsne.tsne(X,2, 100, 15.0);
        
        
        # Generate colors for each class
        colors = []
        for _ in range(len(sorted(set(self.class_indices)))):
            while True:
                r,g,b = random.random(),random.random(),random.random()
                if not (r,g,b) in colors:
                    break
            colors.append((r,g,b)) 
            
        Plot.scatter(Y[:,0], Y[:,1], 1, color = colors);
        show()
        self.__save_plot__(path,'testdata')
    
    def __generate_class_indices__(self,path, batches):
        indices_collected = []
    
        for batch in batches:
            indices_collected += pickle.load( open( path+'/class_indices_batch_'+str(batch)+'.p', "rb" ) )
            
        return indices_collected
            
    def __save_plot__(self,path,name):    
        Plot.savefig(path+'/'+name+'.png')
            
        
