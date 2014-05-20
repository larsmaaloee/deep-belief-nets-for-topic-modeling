__author__ = 'larsmaaloee'

from numpy import *
import numpy.random as rand
import serialization as s
import DataPreparation
import dbn
import matplotlib.pyplot as plt
import env_paths

class PreTraining:
    """
    This class will do the training of the RBM's. The end results will be corrected weights and biases from the pre-
    training.
    """

    def __init__(self,visible_units,hidden_units,batches,layer_index,fout,fprogress):
        """
        Initialize the values for the RBM.
        
        Parameters
        ----------
        @param visible_units: Number of visible units.
        @param hidden_units: Number of hidden units.
        @param batches: A list of batch sizes.
        @param is_first_layer: Boolean if we are in first layer of the DBN.
        """

        # Set the bag of words data.
        self.batches = batches # Set the data generated. Must be in array.
        self.path = env_paths.get_rbm_data_path()
        self.layer_index = layer_index

        # Progress and info monitoring
        self.fout = fout
        self.fprogress = fprogress
        self.error = []
        
        # Set the number of units in each layer: visible and hidden.
        self.num_vis = visible_units # Set the number of visible units
        self.num_hid = hidden_units # Set the number of hidden units
       
        # Following values should be modified to find the best value.
        self.learning_rate = 0.01
        self.weight_cost = 0.0002
        self.momentum = 0.9
        
        
        self.words = 0
           
        # Set the initial weights and biases
        mu, sigma = 0, sqrt(0.01)
        self.weights = random.normal(mu,sigma,(self.num_vis,self.num_hid)) # Initiate Weight matrix from a 0-mean normal distribution with variance 0.01.
        self.delta_weights = zeros((self.num_vis,self.num_hid),dtype = float64) # This is the delta value that needs to be added to the weight matrix after.
        self.hidden_biases = zeros(self.num_hid,dtype = float64) # The biases for the hidden units.
        self.delta_hidden_biases = zeros(self.num_hid,dtype = float64) # The delta biases that needs to be added to the biases.
        self.visible_biases = zeros(self.num_vis,dtype = float64) # The biases for the visible units.
        self.delta_visible_biases = zeros(self.num_vis,dtype = float64) # The delta biases that needs to be added to the biases.
        
        
    def rsm_learn(self,epochs):
        """
        The learning of the first layer RBM (Replicated Softmax Model). The higher value of epochs will result in
        more training.

        @param epochs: The number of epochs.
        """
        for epoch in range(epochs):
            perplexity = 0
            batch_index = 0

            for _ in self.batches:

                # Positive phase - generate data from visible to hidden units.
                pos_vis = self.__get_input_data__(batch_index,first_layer=True)
                batch_size = len(pos_vis)
                D = sum(pos_vis,axis = 1)
                if epoch == 0:
                   self.words += sum(pos_vis) # Calculate the number of words in order to calculate the perplexity.

                pos_hid_prob = dbn.sigmoid(dot(pos_vis,self.weights)+outer(D, self.hidden_biases))
                self.__save_output__(batch_index, pos_hid_prob) # Serialize the output of the RBM

                # If probabilities are higher than randomly generated, the states are 1
                randoms = rand.rand(batch_size,self.num_hid)
                pos_hid = array(randoms < pos_hid_prob,dtype = int)

                # Negative phase - generate data from hidden to visible units and then again to hidden units.
                neg_vis = pos_vis
                neg_hid_prob = pos_hid
                for i in range(100): # There is only 1 step of contrastive divergence
                    neg_vis,neg_hid_prob,D,p = self.__contrastive_divergence_rsm__(neg_vis, pos_hid_prob, D)
                    if i == 0:
                        perplexity+=p

                pos_products = dot(pos_vis.T,pos_hid_prob)
                pos_visible_bias_activation = sum(pos_vis,axis = 0)
                pos_hidden_bias_activation = sum(pos_hid_prob,axis = 0)
                neg_products = dot(neg_vis.T,neg_hid_prob)
                neg_visibe_bias_activation = sum(neg_vis,axis = 0)
                neg_hidden_bias_activation = sum(neg_hid_prob,axis = 0)

                # Update the weights and biases
                self.delta_weights = self.momentum * self.delta_weights + self.learning_rate * ((pos_products-neg_products)/batch_size - self.weight_cost * self.weights)
                self.delta_visible_biases = (self.momentum * self.delta_visible_biases + (pos_visible_bias_activation-neg_visibe_bias_activation))*(self.learning_rate/batch_size)
                self.delta_hidden_biases = (self.momentum * self.delta_hidden_biases + (pos_hidden_bias_activation-neg_hidden_bias_activation))*(self.learning_rate/batch_size)
                self.weights += self.delta_weights
                self.visible_biases += self.delta_visible_biases
                self.hidden_biases += self.delta_hidden_biases
                batch_index += 1

            if not epoch == 0: # Output error score.
                perplexity = exp(-perplexity/self.words)
                err_str = "Epoch[%2d]: Perplexity = %.02f"%(epoch,perplexity)
                self.fout(err_str)
                self.error += [perplexity]
            self.fprogress()


    def __contrastive_divergence_rsm__(self,vis,hid,D):
        neg_vis = dot(hid,self.weights.T)+self.visible_biases
        softmax_value = dbn.softmax(neg_vis)
        neg_vis *= 0
        for i in xrange(len(vis)):
            neg_vis[i] = random.multinomial(D[i],softmax_value[i],size = 1)
        D = sum(neg_vis,axis = 1)

        perplexity = nansum(vis * log(softmax_value))

        neg_hid_prob = dbn.sigmoid(dot(neg_vis,self.weights)+outer(D,self.hidden_biases))

        return neg_vis,neg_hid_prob,D,perplexity

        
    def rbm_learn(self,epochs,first_layer = False,linear = False):
        """
        The learning of the RBMs. The higher value of epochs will result in more training.

        @param epochs: The number of epochs.
        """
        if linear:
            self.learning_rate = self.learning_rate*0.01

        for epoch in range(epochs):
            errsum = 0
            batch_index = 0
            for _ in self.batches:
                # Positive phase - generate data from visible to hidden units.
                pos_vis = self.__get_input_data__(batch_index,first_layer=first_layer)
                batch_size = len(pos_vis)

                if linear:
                    pos_hid_prob = dot(pos_vis,self.weights) + tile(self.hidden_biases,(batch_size,1))

                else:
                    pos_hid_prob = dbn.sigmoid(dot(pos_vis,self.weights) + tile(self.hidden_biases,(batch_size,1)))

                self.__save_output__(batch_index, pos_hid_prob) # Serialize the output of the RBM

                # If probabilities are higher than randomly generated, the states are 1 
                randoms = rand.rand(batch_size,self.num_hid)
                pos_hid = array(randoms < pos_hid_prob,dtype = int)

                # Negative phase - generate data from hidden to visible units and then again to hidden units.
                neg_vis = pos_vis
                neg_hid_prob = pos_hid
                for i in range(1): # There is only 1 step of contrastive divergence
                    neg_vis, neg_hid_prob = self.__contrastive_divergence_rbm__(neg_vis, pos_hid_prob,linear)

                # Set the error
                errsum += sum(((pos_vis)-neg_vis)**2)/len(pos_vis)

                # Update weights and biases
                self.delta_weights = self.momentum * self.delta_weights + self.learning_rate*((dot(pos_vis.T,pos_hid_prob)-dot(neg_vis.T, neg_hid_prob))/batch_size - self.weight_cost*self.weights)# TODO: RE-EVALUATE THE LAST LEARNING RATE
                self.delta_visible_biases = self.momentum * self.delta_visible_biases + (self.learning_rate/batch_size) * (sum(pos_vis,axis = 0)-sum(neg_vis,axis=0))
                self.delta_hidden_biases = self.momentum * self.delta_hidden_biases + (self.learning_rate/batch_size) * (sum(pos_hid_prob,axis = 0)-sum(neg_hid_prob,axis=0))
                self.weights += self.delta_weights
                self.visible_biases += self.delta_visible_biases
                self.hidden_biases += self.delta_hidden_biases
                batch_index += 1

            # Output error scores
            e = errsum/len(self.batches)
            err_str = "Epoch[%2d]: Error = %.07f"%(epoch+1,e)
            self.fout(err_str)
            self.error += [e]
            self.fprogress()

    def __contrastive_divergence_rbm__(self,vis,hid,linear):
        neg_vis = dbn.sigmoid(dot(hid,self.weights.T) + tile(self.visible_biases,(len(vis),1)))
        if linear:
            neg_hid_prob = dot(neg_vis,self.weights) + tile(self.hidden_biases,(len(vis),1))
        else:
            neg_hid_prob = dbn.sigmoid(dot(neg_vis,self.weights) + tile(self.hidden_biases,(len(vis),1)))
        return neg_vis,neg_hid_prob
    

    def __get_input_data__(self,batch_index,first_layer):
        """
        Retrieve the word-count matrix from HDD.

        @param batch_index: Index of the batch.

        @return: The word-count matrix corresponding to the batch_index.
        """
        if first_layer:
            return DataPreparation.data_processing.get_bag_of_words_matrix(self.batches[batch_index])
        return array(s.load(open(env_paths.get_rbm_output_path(self.num_vis,batch_index,self.layer_index-1),"rb")))
    
    def __save_output__(self,batch_index,outputs):
        """
        Serialize the output of the rbm.
        
        @param batch_index: Index of the batch.
        @param outputs: The output probabilitites of the rbm
        """
        
        s.dump(outputs.tolist() , open(env_paths.get_rbm_output_path(str(self.num_hid),batch_index,self.layer_index), "wb" ) )
        
    def __plot_development__(self,pos_vis,neg_vis,epoch):
        fig = plt.figure()
        fig.hold()
        plt.plot([i for i in range(500)],pos_vis)
        plt.savefig(env_paths.get_rbm_plotting_input(epoch,self.num_vis))
        fig = plt.figure()
        fig.hold()
        plt.plot([i for i in range(500)],pos_vis)
        plt.plot([i for i in range(500)],neg_vis)
        plt.savefig(env_paths.get_rbm_plotting_output(epoch,self.num_vis))
        plt.close()
