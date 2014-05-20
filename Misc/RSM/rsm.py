'''
Created on Sep 11, 2013

@author: larsmaaloee
'''

from numpy import *
import numpy.random as rand
import os
import cPickle as pickle
from DataPreparation import data_processing

class RSM(object):
    
    def __init__(self,visible_units,hidden_units,batches):
        '''
        Initialize the values for the restricted boltzmann machine.
        
        Parameters
        ----------
        num_vis: Number of visible units.
        num_hid: Number of hidden units.
        batches: Batch sizes corresponding to the bag of words.
        is_first_layer: Boolean if we are in first layer.
        is_final_layer: Boolean if we are in the final layer.
        save_rbm: Should the data be saved.
        '''
       
        
        self.batches = batches # Set the data generated. Must be in array.
        
        
        # Set the number of units in each layer: visible and hidden.
        self.num_vis = visible_units # Set the number of visible units
        self.num_hid = hidden_units # Set the number of hidden units
       
        # Following values should be modified to find the best value.
        self.epsilon_weights = 0.01 # Learning rate for weights
        self.epsilon_visibleBiases = 0.01 # Learning rate for visible biases
        self.epsilon_hiddenBiases = 0.01 # Learning rate for hidden biases
           
        self.momentum = 0.9 # Initialize the learning momentum
        
        # Set the initial weights and biases
        #self.weights = standard_normal((self.num_vis,self.num_hid))/100 # Initiate Weight matrix from a 0-mean normal distribution with variance 0.01.
        self.weights = self.epsilon_weights*random.randn(self.num_vis, self.num_hid)
        self.delta_weights = zeros((self.num_vis,self.num_hid),dtype = float64) # This is the delta value that needs to be added to the weight matrix after. 
        self.hidden_biases = zeros(self.num_hid,dtype = float64) # The biases for the hidden units.
        self.delta_hidden_biases = zeros(self.num_hid,dtype = float64) # The delta biases that needs to be added to the biases.
        self.visible_biases = zeros(self.num_vis,dtype = float64) # The biases for the visible units.
        self.delta_visible_biases = zeros(self.num_vis,dtype = float64) # The delta biases that needs to be added to the biases.
        
        
        
    def rsm_learn(self,epochs):
        '''
        Learning method for the replicated softmax.
        The higher value of epochs will result in
        more training.
        
        Parameters
        ----------
        epochs: The number of epochs.
        '''
        
        for epoch in range(epochs):
            errsum = 0
            batch_index = 0
            for _ in self.batches:
                
                # Positive phase - generate data from visible to hidden units.
                pos_vis = data_processing.get_bag_of_words_matrix(self.batches[batch_index])
                D = sum(pos_vis,axis = 1)
                batch_size = len(pos_vis)
                
                #pos_hid_prob = (1+sp.tanh((dot(pos_vis,self.weights)+outer(D, self.hidden_biases))/2))/2
                pos_hid_prob = sigmoid(dot(pos_vis,self.weights)+outer(D, self.hidden_biases))        
                
                # If probabilities are higher than randomly generated, the states are 1 
                randoms = rand.rand(batch_size,self.num_hid)
                pos_hid = array(randoms < pos_hid_prob,dtype = int)
                
                # Negative phase - generate data from hidden to visible units and then again to hidden units.                
                neg_vis = dot(pos_hid,self.weights.T)+self.visible_biases
                tmp = exp(neg_vis)
                s = tmp.sum(axis = 1)
                s = s.reshape((batch_size,1))
                neg_vis_pdf = tmp/s
                
                neg_vis *= 0
                for i in xrange(batch_size):
                    neg_vis[i] = random.multinomial(D[i],neg_vis_pdf[i],size = 1)
                        
                                       
                neg_hid_prob = sigmoid(dot(neg_vis,self.weights)+outer(D,self.hidden_biases))
                #neg_hid_prob = (1+sp.tanh((dot(neg_vis,self.weights)+outer(D,self.hidden_biases))/2))/2
                
                # Set the error
                errsum += sum(((pos_vis)-neg_vis)**2)
                
                self.delta_weights = self.delta_weights*self.momentum + dot(pos_vis.T, pos_hid_prob) - dot(neg_vis.T, neg_hid_prob)
                self.delta_visible_biases = self.delta_visible_biases * self.momentum + pos_vis.sum(axis = 0) - neg_vis.sum(axis = 0)
                self.delta_hidden_biases = self.delta_hidden_biases * self.momentum + pos_hid_prob.sum(axis = 0) - neg_hid_prob.sum(axis = 0)
                
                self.weights += self.delta_weights * (self.epsilon_weights/batch_size)
                self.visible_biases += self.delta_visible_biases * (self.epsilon_visibleBiases/batch_size)
                self.hidden_biases += self.delta_hidden_biases * (self.epsilon_hiddenBiases/batch_size)
                
                batch_index += 1
            print 'Epoch ',epoch+1,' Error ',errsum/batch_size
        
        self.__save_rsm__()
        
    
    def __save_rsm__(self):
        path = 'pickle/rsm'
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(self.weights , open( path+"/weight_matrices.p", "wb" ) )
        pickle.dump(self.hidden_biases , open( path+"/hidden_biases.p", "wb" ) )
        pickle.dump(self.visible_biases , open( path+"/visible_biases.p", "wb" ) )

    

def sigmoid(x):
    #return (1 + sp.tanh(x/2))/2
    return 1./(1+exp(-x))

def get_weights():
    return pickle.load( open( 'pickle/rsm/weight_matrices.p', "rb" ) )

def get_visible_biases():
    return pickle.load( open( 'pickle/rsm/visible_biases.p', "rb" ) )

def get_hidden_biases():
    return pickle.load( open( 'pickle/rsm/hidden_biases.p', "rb" ) )


def generate_output_data(d, weights, visible_biases, hidden_biases):
        '''
        Run through the RBM and
        compute the output.
        '''
        vis = d
        D = sum(vis,axis = 1)
        batch_size = len(vis)
        #pos_hid_prob = (1+sp.tanh((dot(vis,self.weights)+outer(D, self.hidden_biases))/2))/2
        pos_hid_prob = sigmoid(dot(vis,weights)+outer(D, hidden_biases))        
        # If probabilities are higher than randomly generated, the states are 1 
        randoms = rand.rand(batch_size,len(hidden_biases))
        hid = array(randoms < pos_hid_prob,dtype = int)
        hid = pos_hid_prob
        return hid