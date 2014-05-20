__author__ = 'larsmaaloee'

from time import time
from multiprocessing import Process, Pool
from numpy import *
import matplotlib.pyplot as plt
from images2gif import writeGif
from PIL import Image
from DBN.pretraining import PreTraining
import serialization as s
from DBN.finetuning import FineTuning
from DataPreparation import data_processing
import env_paths


class DBN:
    """
    This class will generate the DBN. It will train the RBM's in the given network - pretraining. Afterwards it
    will train the Deep Autoencoder - finetuning.
    """

    def __init__(self,visible_units,batches,hidden_layers,output_units,max_epochs,plot = False,binary_output = False):
        """
        Initialize variables of the DBN.

        @param visible_units: Number of visible units in the DBN.
        @param batches: The list of batches of the training set.
        @param hidden_layers: A list containing numbers of neurons in each of the hidden layers. i.e. [500,250,125].
        @param output_units: Number of output units.
        @param max_epochs: The number of epochs of pretraining and finetuning.
        @param plot: If plots must be computed during training.
        @param binary_output: If the DBN should be computed to generate binary output.
        """
        self.visible_units = visible_units
        self.batches = batches
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.max_epochs = max_epochs
        self.weight_matrices = []
        self.hidden_biases = []
        self.visible_biases = []
        self.binary_output = binary_output
        
        # Progress parameters. Used for progress bar in web framework.
        self.output_txt = []
        self.progress_goal = float((2+len(hidden_layers)-1)*max_epochs+max_epochs)
        self.progress = 0.
        
        self.plot_dic = {}
        if plot:
            self.plot = True
        else:
            self.plot = False
            
    
    def run_dbn(self):
        """
        Run the pretraining and the finetuning of the DBN.
        """
        self.run_pretraining()
        self.run_finetuning()
        
    
    def run_pretraining(self):
        """
        Run the pretraining of the DBN. Each RBM will be trained from bottom to top of the DBN.
        """
        # Following code is the initialisation of a dummy process in order to allow numpy and multiprocessing to run
        # properly on OSX.
        pretraining_process = Process(target = self.__run_pretraining_as_process,args = ())
        pretraining_process.start()
        pretraining_process.join()

    def __run_pretraining_as_process(self):
        rbm_index = 0
        self.print_output('Pre Training')
        timer = time()
        # Bottom layer
        self.print_output('Visible units: '+str(self.visible_units)+' Hidden units: '+str(self.hidden_layers[0]))
        r = PreTraining(self.visible_units,self.hidden_layers[0],self.batches,rbm_index,self.print_output,self.increment_progress)
        r.rsm_learn(self.max_epochs)
        self.plot_dic[self.visible_units] = r.error
        self.weight_matrices.append(r.weights)
        self.hidden_biases.append(r.hidden_biases)
        self.visible_biases.append(r.visible_biases)
        rbm_index += 1
        # Middle layers
        for i in range(len(self.hidden_layers)-1):
            self.print_output('Top units: '+str(self.hidden_layers[i])+' Bottom units: '+str(self.hidden_layers[i+1]))
            r = PreTraining(self.hidden_layers[i],self.hidden_layers[i+1],self.batches,rbm_index,self.print_output,self.increment_progress)
            r.rbm_learn(self.max_epochs)
            self.plot_dic[self.hidden_layers[i]] = r.error
            self.weight_matrices.append(r.weights)
            self.hidden_biases.append(r.hidden_biases)
            self.visible_biases.append(r.visible_biases)
            rbm_index += 1
        # Top layer
        self.print_output('Top units: '+str(self.hidden_layers[len(self.hidden_layers)-1])+' Output units: '+str(self.output_units))
        r = PreTraining(self.hidden_layers[len(self.hidden_layers)-1],self.output_units,self.batches,rbm_index,self.print_output,self.increment_progress)
        r.rbm_learn(self.max_epochs,linear = True)
        self.plot_dic[self.hidden_layers[-1]] = r.error
        self.weight_matrices.append(r.weights)
        self.hidden_biases.append(r.hidden_biases)
        self.visible_biases.append(r.visible_biases)
        print 'Time ',time()-timer
        # Save the biases and the weights.
        save_rbm_weights(self.weight_matrices,self.hidden_biases,self.visible_biases)
        self.save_output(finetuning=False)
        # Plot
        if self.plot:
            self.generate_gif_rbm()

    def run_finetuning(self, load_from_serialization = False):
        """
        Run the finetuning of the DBN. This will only run if the pretraining has been run.

        @param load_from_serialization: If the weight matrices are not loaded into memory, this should be True.
        """
        try:
            if load_from_serialization:
                self.weight_matrices,self.hidden_biases,self.visible_biases = load_rbm_weights()
        except IOError:
            print 'Please run pretraining before executing finetuning.'
            return

        self.print_output('Fine Tuning')
        timer = time()

        fine_tuning = FineTuning(self.weight_matrices, self.batches,self.print_output,self.increment_progress,
                                 self.hidden_biases,self.visible_biases,binary_output = self.binary_output)
        fine_tuning.run_finetuning(self.max_epochs)
        
        fine_tuning_error_train = fine_tuning.train_error
        fine_tuning_error_test = fine_tuning.test_error
        if self.plot:
            self.plot_finetuning_error(fine_tuning_error_train, fine_tuning_error_test)

        self.print_output('Time '+str(time()-timer))
        save_dbn(fine_tuning.weight_matrices_added_biases,fine_tuning_error_train,fine_tuning_error_test)
        self.output_dbn_errors(fine_tuning_error_train,fine_tuning_error_test)
        self.save_output()

    def continue_finetuning(self,epochs,binary_output = False):
        """
        Continue finetuning. This will only run if the finetuning has already run.

        @param epochs: Number of epochs to run the finetuning.
        @param binary_output: If the DBN must output binary values.
        """

        self.weight_matrices = load_dbn_weights()
        self.load_finetuning_output_txt()
        self.print_output('Fine Tuning (continued)')
        timer = time()
        fine_tuning = FineTuning(self.weight_matrices, self.batches,self.print_output,self.increment_progress,binary_output = binary_output)
        
        fine_tuning.run_finetuning(epochs)
        
        fine_tuning_error_train = fine_tuning.train_error
        fine_tuning_error_test = fine_tuning.test_error
        
        self.print_output('Time '+str(time()-timer))
        save_dbn(fine_tuning.weight_matrices_added_biases,fine_tuning_error_train,fine_tuning_error_test)
        self.output_dbn_errors(fine_tuning_error_train,fine_tuning_error_test)
        self.save_output()

    def output_dbn_errors(self,fine_tuning_error_train,fine_tuning_error_test):
        """
        @param fine_tuning_error_train: Dictionary of the train error for each epoch.
        @param fine_tuning_error_test: Dictionary of the test error for each epoch.
        """
        self.print_output("Finetuning train error:")
        for k in fine_tuning_error_train:
            self.print_output("Train error epoch["+str(k+1)+"]: "+str(fine_tuning_error_train[k]))
        self.print_output("Finetuning test error:")
        for k in fine_tuning_error_test:
            self.print_output("Test error epoch["+str(k+1)+"]: "+str(fine_tuning_error_test[k]))

    def save_output(self, finetuning = True):
        """
        Save the output of the progress of the training process for the pretraining or the finetuning.
        @param finetuning: If finetuning is True, the error for finetuning should be saved and vice versa.
        """
        p = env_paths.get_dbn_output_txt_path() if finetuning else env_paths.get_rbm_output_txt_path()
        out = open(p,"w")
        for elem in self.output_txt:
            out.write(elem+"\n")
        out.close()
        self.output_txt = []

    def load_finetuning_output_txt(self):
        txt = open(env_paths.get_dbn_output_txt_path(),"r")
        while True:
            line = txt.readline()
            if not line:
                break
            self.output_txt.append(line)

    def print_output(self,txt = None):
        if not txt == None:
            print txt
            self.output_txt.append(txt)
        return self.output_txt

    def increment_progress(self):
        self.progress += 1.
        
    def get_progress(self):
        return (self.progress/self.progress_goal)*100.
    
    def get_output(self):
        return self.output_txt
    
    def plot_finetuning_error(self,train_error,test_error):
        path = env_paths.get_web_server_img_path()
        f = plt.figure()
        f.hold()
        plt.plot(train_error.values())
        plt.plot(test_error.values())
        plt.savefig(path+"/dbn.png")
        
    def generate_gif_rbm(self):
        path = env_paths.get_web_server_img_path()
        self.plot_rbm(path)
        
        paths = os.listdir(path)
        file_names = []
        for p in paths:
            if 'rbm' in p:
                file_names.append(path+p)
        
        images = [Image.open(fn) for fn in file_names]            
        filename = path+"/rbm.GIF"
        writeGif(filename, images, duration=5.0)

    def plot_rbm(self,path):
        count = 0
        for layer in sorted(self.plot_dic): 
            f = plt.figure()
            f.hold()
            plt.title('Layer '+str(layer))
            plt.plot(self.plot_dic[layer])
            plt.savefig(path+"/rbm_"+str(count)+".png")
            count += 1


def save_rbm_weights(weight_matrices,hidden_biases,visible_biases):
    """
    Save the weight matrices from the rbm pretraining.

    @param weight_matrices: the weight matrices of the rbm pretraining.
    """
    s.dump([w.tolist() for w in weight_matrices] , open( env_paths.get_rbm_weights_path(), "wb" ) )
    s.dump([b.tolist() for b in hidden_biases] , open( env_paths.get_rbm_hidden_biases_path(), "wb" ) )
    s.dump([b.tolist() for b in visible_biases] , open( env_paths.get_rbm_visible_biases_path(), "wb" ) )

def save_dbn(weight_matrices,fine_tuning_error_train,fine_tuning_error_test,output = None):
    """
    Save the deep belief network into serialized files.

    @param weight_matrices: the weight matrices of the deep belief network.
    """
    s.dump([w.tolist() for w in weight_matrices] , open( env_paths.get_dbn_weight_path(), "wb" ) )
    s.dump(fine_tuning_error_train , open( env_paths.get_dbn_training_error_path(), "wb" ) )
    s.dump(fine_tuning_error_test , open( env_paths.get_dbn_test_error_path(), "wb" ) )

    if not output == None:
        out = open(env_paths.get_dbn_output_txt_path(),"w")
        for elem in output:
            out.write(elem+"\n")
        out.close()

def load_dbn_weights():
    """
    Load the weight matrices from the finetuning.

    @param weight_matrices: the weight matrices of the finetuning.
    """
    return [array(w) for w in s.load(open(env_paths.get_dbn_weight_path(), "rb" ) )]


def load_rbm_weights():
    """
    Load the weight matrices from the rbm pretraining.

    @param weight_matrices: the weight matrices of the rbm pretraining.
    """
    weights = [array(w) for w in s.load( open( env_paths.get_rbm_weights_path(), "rb" ) )]
    hid_bias = [array(b) for b in s.load( open( env_paths.get_rbm_hidden_biases_path(), "rb" ) )]
    vis_bias = [array(b) for b in s.load( open( env_paths.get_rbm_visible_biases_path(), "rb" ) )]
    return weights,hid_bias,vis_bias

def sigmoid(x):
    return 1./(1+exp(-x))

def softmax(x):
    numerator = exp(x)
    denominator = numerator.sum(axis = 1)
    denominator = denominator.reshape((x.shape[0],1))
    softmax = numerator/denominator
    return softmax

def get_weights():
    """
    Retrieve the weights from the generated DBN.

    @return: Weights of the DBN.
    """
    return [array(w) for w in s.load(open(env_paths.get_dbn_weight_path(), "rb" ) )]

def generate_output_for_test_data(binary_output = False):
    """
    For all test data, generate the output and add to a list.

    @return: List of all output data.
    """

    weight_matrices_added_biases = get_weights()
    batches = data_processing.get_batch_list(training = False)
    output_data = []

    evaluations = []
    for batch in range(len(batches)):
        evaluations.append((batches[batch],weight_matrices_added_biases,binary_output))
    p = Pool(6)
    results = p.map(__generate_output_for_test_data_par,evaluations)
    p.close()
    p.join()
    for elem in results:
        output_data+=elem
    return output_data


def __generate_output_for_test_data_par(args):
    batch,weight_matrices_added_biases,binary_output = args
    d = data_processing.get_bag_of_words_matrix(batch,training = False)
    return list((generate_output_data(d, weight_matrices_added_biases,binary_output)))


def generate_output_for_train_data(binary_output = False):
    """
    For all train data, generate the output and add to a list.

    @return: List of all output data.
    """
    weight_matrices_added_biases = get_weights()
    batches = data_processing.get_batch_list(training = True)
    output_data = []
    
    evaluations = []
    for batch in range(len(batches)):
        evaluations.append((batches[batch],weight_matrices_added_biases,binary_output))
    p = Pool(6)
    results = p.map(__generate_output_for_train_data_par,evaluations)
    p.close()
    p.join()
    for elem in results:
        output_data+=elem
    return output_data

def __generate_output_for_train_data_par(args):
    batch,weight_matrices_added_biases,binary_output = args
    d = data_processing.get_bag_of_words_matrix(batch,training = True)
    return list((generate_output_data(d, weight_matrices_added_biases,binary_output=binary_output)))

def generate_input_data_list(training = True):
    """
    Generate a list of all input data.

    @param training: If training is True, the input should be generated for training data and vice versa.
    """
    batches = data_processing.get_batch_list(training = training)
    input_data = []

    for batch in range(len(batches)):
        print 'Batch ',batch + 1, ' of ',len(batches)
        d = data_processing.get_bag_of_words_matrix(batches[batch],training=training)
        d = get_norm_x(d)
        input_data += list(d)

    return input_data

def get_norm_x(x_matrix):
    """
    Normalize the BOW matrix and make sure not to do division by 0.
    @param x_matrix: The BOW matrix.
    @return: The normalized BOW.
    """
    sum_x = sum(x_matrix,axis = 1)
    indices = where(sum_x == 0)
    for i in indices:
        sum_x[i] = 1.
    norm_x = x_matrix/sum_x[newaxis].T
    return norm_x

def generate_output_data(x, weight_matrices_added_biases,binary_output=False):
    """
    Generate a forwards-pass through the DBN.

    @param x: BOW matrix for a batch.
    @param weight_matrices_added_biases: The weight matrices where the biases are added to the -1 row.
    """
    if binary_output:
        threshold = 0.1
    output_batch = []
    for d in x:
        z_values = []
        for i in range(len(weight_matrices_added_biases)/2):
            if i == 0:
                z = sigmoid(dot(d,weight_matrices_added_biases[i][:-1,:])+outer(sum(d), weight_matrices_added_biases[i][-1,:]))
            elif i == (len(weight_matrices_added_biases)/2)-1:
                z = dot(z_values[i-1],weight_matrices_added_biases[i])
            else:
                z = sigmoid(dot(z_values[i-1],weight_matrices_added_biases[i]))

            z = append(z,ones(1,dtype = float64))
            z_values.append(z)

        if binary_output: out = array(z[:-1]<=threshold,dtype = int)
        else: out = z[:-1]
        output_batch.append(out)
    return array(output_batch)