__author__ = 'larsmaaloee'

import os
import env_paths as ep
import matplotlib
import numpy as np
from DBN.dbn import generate_output_for_test_data, generate_output_for_train_data, generate_input_data_list
from DataPreparation.data_processing import get_all_class_indices, get_all_class_names
from heapq import nsmallest
from scipy.spatial.distance import cosine, euclidean, cdist
from multiprocessing import Pool
import multiprocessing
import serialization as s
from collections import Counter
from sklearn.metrics import confusion_matrix
import pylab as plot
import time

class DBNTesting:
    def __init__(self,testing = True,image_data = False,binary_output = False):
        """
        @param testing: Should be True if test data is to be plottet. Otherwise False.
        @param image_data: If the testing should be done on image data.
        @param binary_output: If the output of the DBN must be binary.
        """
        if not check_for_data:
            print 'No DBN data or testing data.'
            return

        self.status = -1
        self.output = []
        self.testing = testing
        self.image_data = image_data
        self.binary_output = binary_output

        try:
            self.output_data = s.load(open('output/output_data.p','rb'))
            self.class_indices = s.load(open('output/class_indices.p','rb'))
        except:
            self.output_data = generate_output_for_test_data(image_data=self.image_data,binary_output=self.binary_output) if testing else generate_output_for_train_data(
                image_data=self.image_data,binary_output=self.binary_output)
            self.class_indices = get_all_class_indices(training = False) if testing else get_all_class_indices()
            s.dump([out.tolist() for out in self.output_data],open('output/output_data.p','wb'))
            s.dump(self.class_indices,open('output/class_indices.p','wb'))

        self.output_data = np.array(self.output_data)

    def generate_accuracy_measurement(self,evaluation_points):
        """
        Generate an accuracy measurement for the current DBN. This method will run through each output of the
        dataset and check whether its X neighbors are of the same category. The amount of neighbors will evalu-
        ate in a percentage score. So for instance an output who has 3 neighbors where 2 are of the same cate-
        gory will get the accuracy score of 2/3. All accuracy scores are averaged at the end. This algorithm will
        run for an X amound of evaluation_points.

        @param evaluation_points: A list containing the number of neighbors that are to be evaluated. i.e. [1,3]
        means that the method should calculate the accuracy measurement for 1 and 3 neighbors.
        """
        accuracies = []
        for e in evaluation_points:
            self.__output('Evaluation: '+str(e))
            acc = 0.0
            now = time.time()
            for it in range(len(self.output_data)):
                o1 = self.output_data[it]
                if self.binary_output:
                    distances = np.array(hamming_distance(o1,self.output_data),dtype = float)
                    distances[it] = np.Inf
                else:
                    distances = np.array(distance(o1,self.output_data),dtype = float)
                    distances[it] = np.inf

                # Retrieve the indices of the n maximum values
                minimum_values = nsmallest(e, distances)

                indices = []
                for m in minimum_values:
                    i = list(np.where(np.array(distances)==m)[0])
                    indices += i

                acc_temp = 0.0
                for i in indices:
                    if self.class_indices[i] == self.class_indices[it]:
                        acc_temp += 1.0
                acc_temp /= len(indices)
                acc += acc_temp
                if it+1 % 1000 == 0:
                    print 'Time: ',time.time()-now
                    now = time.time()
                    self.__output('Correct: '+str((acc/(it+1))*100)[:4]+"%"+' of '+str(it+1))
            accuracies.append(acc/len(self.output_data))
        for i in range(len(accuracies)):
            self.__output("Eval["+str(evaluation_points[i])+"]: "+str(accuracies[i]*100)+"%")
        self.__write_output_to_file()

    def __write_output_to_file(self):
        print 'Outputting test scores to output folder.'
        f = open('output/testscores.txt','wb')

        for i in range(len(self.output)):
            s = self.output[i]
            f.write(s+"\n")
        f.close()

    def __output(self,s):
        f = open('output/testscores.txt','a')
        f.write(s+"\n")
        f.close()
        print s
        self.output.append(s)

    def generate_accuracy_measurement_parallel(self,evaluation_points):
        """
        Parallel implementation of the accuracy measurement.
        Generate an accuracy measurement for the current DBN. This method will run through each output of the
        dataset and check whether its X neighbors are of the same category. The amount of neighbors will evalu-
        ate in a percentage score. So for instance an output who has 3 neighbors where 2 are of the same cate-
        gory will get the accuracy score of 2/3. All accuracy scores are averaged at the end. This algorithm will
        run for an X amound of evaluation_points.

        @param evaluation_points: A list containing the number of neighbors that are to be evaluated. i.e. [1,3]
        means that the method should calculate the accuracy measurement for 1 and 3 neighbors.
        """

        # Split outpudata for multiprocessing purposes.
        self.split_output_data = []
        self.split_class_indices = []
        tmp_output_data = []
        tmp_class_indices = []
        for i in xrange(len(self.output_data)):
            if i>0 and i % 100 == 0:
                self.split_output_data.append(tmp_output_data)
                self.split_class_indices.append(tmp_class_indices)
                tmp_output_data = []
                tmp_class_indices = []
            tmp_output_data.append(self.output_data[i])
            tmp_class_indices.append(self.class_indices[i])

        if len(tmp_output_data) > 0:
            self.split_output_data.append(tmp_output_data)
            self.split_class_indices.append(tmp_class_indices)

        manager = multiprocessing.Manager()
        distances_dict = manager.dict()

        accuracies = []
        for e in evaluation_points:
            self.__output('Evaluation: '+str(e))
            acc = 0.0
            processed = 0
            for i in xrange(len(self.split_output_data)):
                now = time.time()
                o = self.split_output_data[i]
                # init multiprocessing
                manager = multiprocessing.Manager()
                result_queue = manager.Queue()
                p = Pool(6)
                p.map_async(generate_acc_for_doc,[(distances_dict,e,processed+i,result_queue,self.output_data,
                                                   self.class_indices,self.binary_output) for i in range(len(o))])
                p.close()
                p.join()
                print 'time: ',time.time() - now

                for _ in range(len(o)):
                    acc += result_queue.get()
                processed += len(o)
                if processed % 1000 == 0:
                    self.__output('Correct: '+str((acc/(processed))*100)[:4]+"%"+' of '+str(processed))

            accuracies.append(acc/len(self.output_data))


        for i in range(len(accuracies)):
            self.__output("Eval["+str(evaluation_points[i])+"]: "+str(accuracies[i]*100)+"%")

        self.__write_output_to_file()

    def confusion_matrix(self,no_of_neighbors):
        evaluated_class_indices = self.k_nearest_neibors(no_of_neighbors)
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
        matplotlib.rc('font', **font)

        lbls = get_all_class_names()
        for i in range(len(lbls)):
            if len(lbls[i])>12:
                str = lbls[i][:10]
                lbls[i] = str+"..."
        cm = confusion_matrix(self.class_indices,evaluated_class_indices,sorted(set(self.class_indices)))
        # Show confusion matrix
        fig = plot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        ax.set_xticklabels(['']+lbls)
        plot.xticks(rotation = 55)
        ax.set_yticklabels(['']+lbls)
        plot.yticks(rotation = 55)
        plot.ylabel('True label')
        plot.xlabel('Predicted label')
        plot.show()



    def k_nearest_neibors(self,no_of_neighbors):
        evaluated_class_indices = []
        for it in range(len(self.output_data)):
        #for it in range(5):

            o1 = self.output_data[it]
            if self.binary_output:
                distances = np.array(hamming_distance(o1,self.output_data),dtype = float)
                distances[it] = np.Inf
            else:
                # Compute distances between o1 and remaining outputs
                distances = np.array(distance(o1,self.output_data),dtype = float)
                distances[it] = np.inf

            # Retrieve the indices of the n maximum values
            minimum_values = nsmallest(no_of_neighbors, distances)

            indices = []
            for m in minimum_values:
                i = list(np.where(np.array(distances)==m)[0])
                indices += i

            #if len(indices) > e: # TODO: Need to implement method to check for indices with same distance
            #    print 'Indices ',len(indices)-e,' greater than e.'

            c = Counter(indices)
            best_idx = -1
            class_idx = -1
            for i in range(len(c.keys())):
                if c[c.keys()[i]]>c[best_idx]:
                    best_idx = c.keys()[i]
                    class_idx = self.class_indices[best_idx]

            evaluated_class_indices.append(class_idx)

            if it % 1000 == 0 and not it == 0:
                print "Processed %d"%it
        return evaluated_class_indices




def check_for_data():
    """
    Check for DBN network data.
    """
    if not (os.path.exists(ep.get_test_data_path()) or os.path.exists(ep.get_dbn_weight_path())):
        return False
    return True

def distance(v,m):
    #v_tiled = np.tile(v,(len(m),1))
    return cdist(np.array([v]),m,'euclidean')[0]

def hamming_distance(v,m):
    return np.sum((v != m),axis = 1)

def generate_acc_for_doc(args):
    """
    Generate accuracy measurement for a single doc. This function is used as a supplement to the parallel acc. meas-
    urement method.
    """
    distances_dict,e, idx, queue, output_data, class_indices, binary_output = args

    try:
        distances = distances_dict[idx]
    except KeyError:
        o1 = output_data[idx]
        if binary_output:
            distances = np.array(hamming_distance(o1,output_data),dtype = float)
            distances[idx] = np.Inf

        else:
            distances = np.array(distance(o1,output_data),dtype = float)
            distances[idx] = np.inf
        distances_dict[idx] = distances

    # Retrieve the indices of the n smallest values
    minimum_values = nsmallest(e, distances)

    indices = []
    for m in minimum_values:
        i = list(np.where(np.array(distances)==m)[0])
        indices += i

    acc_temp = 0.0
    for i in indices:
        if class_indices[i] == class_indices[idx]:
            acc_temp += 1.0
    acc_temp /= len(indices)

    queue.put(acc_temp)

def LDA_DBN_comparison(lda_output_data,lda_doc_names,dbn_output_data,dbn_doc_names,evaluation_points,binary_output = False):

    dbn_output_data = np.array(dbn_output_data)
    lda_output_data = np.array(lda_output_data)
    # Split outpudata for multiprocessing purposes.
    split_output_data = []
    tmp_output_data = []
    for i in xrange(len(dbn_output_data)):
        if i>0 and i % 100 == 0:
            split_output_data.append(tmp_output_data)
            tmp_output_data = []
        tmp_output_data.append(dbn_output_data[i])

    if len(tmp_output_data) > 0:
        split_output_data.append(tmp_output_data)


    accuracies = []
    for e in evaluation_points:
        print 'Evaluation: ',e
        __append_output_to_file('Evaluation: '+str(e))
        acc = 0.0
        processed = 0
        for i in xrange(len(split_output_data)):
            o = split_output_data[i]
            # init multiprocessing
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()
            p = Pool(6)
            p.map_async(generate_comparison,[(e,lda_output_data,lda_doc_names,dbn_output_data,dbn_doc_names,processed+j,result_queue,binary_output) for j in range(len(o))])
            p.close()
            p.join()

            for _ in range(len(o)):
                acc += result_queue.get()
            processed += len(o)
            if processed % 1000 == 0:
                s = 'Correct: '+str((acc/(processed))*100)[:4]+"%"+' of '+str(processed)
                print s
                __append_output_to_file(s)

        accuracies.append(acc/len(dbn_output_data))

def generate_comparison(args):
    e,lda_output_data,lda_doc_names,dbn_output_data,dbn_doc_names,idx,queue,binary_output = args

    o1 = dbn_output_data[idx]
    dbn_indices = generate_proximity_indices(o1,dbn_output_data,idx,e,binary_output)
    dbn_proximity_names = []
    for i in dbn_indices:
        dbn_proximity_names.append(dbn_doc_names[i])

    dbn_doc_name = dbn_doc_names[idx]
    lda_idx = np.where(np.array(lda_doc_names) == dbn_doc_name)[0][0]
    o1 = lda_output_data[lda_idx]
    lda_indices = generate_proximity_indices(o1,lda_output_data,idx,e,binary_output)
    lda_proximity_names = []
    for i in lda_indices:
        lda_proximity_names.append(lda_doc_names[i])
    # Compare the docnames
    acc_temp = 0.0
    for dn in dbn_proximity_names:
        if dn in lda_proximity_names:
            acc_temp += 1.0

    if acc_temp > e:
        acc_temp = float(e)

    acc_temp /= e
    queue.put(acc_temp)

def generate_proximity_indices(o1,output_data,idx,e,binary_output):
    # Compute distances between o1 and remaining outputs
    distances = distance(o1,output_data)
    distances[idx] = np.inf

    # Retrieve the indices of the n smallest values
    minimum_values = nsmallest(e, distances)

    indices = []
    for m in minimum_values:
        i = list(np.where(np.array(distances)==m)[0])
        indices += i

    return indices

def __write_output_to_file(s):
    f = open('output/testscores.txt','wb')
    f.write(s+"\n")
    f.close()

def __append_output_to_file(s):
    f = open('output/testscores.txt','a')
    f.write(s+"\n")
    f.close()


