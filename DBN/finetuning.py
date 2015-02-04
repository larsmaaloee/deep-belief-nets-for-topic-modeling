__author__ = 'larsmaaloee'

from multiprocessing import Process, Pool, Manager, cpu_count
import os

from numpy import *

from DBN.minimize import minimize
from DataPreparation import data_processing
from DataPreparation.data_processing import get_bag_of_words_matrix
import dbn
import serialization as s
import env_paths


class FineTuning:
    def __init__(self, weight_matrices, batches, fout, hidden_biases=None, visible_biases=None, binary_output=False):
        """
        Initialize variables of the finetuning.

        @param weight_matrices: The list of weight matrices for the DBN.
        @param batches: The list of batch-sizes.
        @param fout: The output function. For progress monitoring of the training.
        @param hidden_biases: The hidden biases for the finetuning.
        @param visible_biases: The visible biases for the finetuning.
        @param binary_output: If the output of the DBN must be binary. If so, Gaussian noise will be added to bottleneck.
        """

        # Progress and info monitoring
        self.fout = fout
        self.binary_output = binary_output
        # Generate sampled noise matrices
        self.sampled_noise = []
        self.current_sampled_noise = None

        self.batches = batches
        if hidden_biases == None:  # If finetuning shall continue.
            self.weight_matrices_added_biases = weight_matrices
        else:
            weight_matrices = weight_matrices
            hidden_biases = hidden_biases
            visible_biases = visible_biases

            # Unfold network to make deep autoencoder
            self.weight_matrices_added_biases = []
            weight_matrices = append(weight_matrices, weight_matrices[::-1])
            visible_hidden_biases = append(hidden_biases, visible_biases[::-1])
            # Add the visible and hidden biases to the weight matrices
            for i in range(len(weight_matrices)):
                if i < (len(weight_matrices) / 2):
                    # Make sure that the hidden biases are in the same format as the weight matrix
                    tmp = zeros((1, len(visible_hidden_biases[i])))
                    tmp[0] = visible_hidden_biases[i]
                    self.weight_matrices_added_biases.append(append(weight_matrices[i], tmp, axis=0))
                else:
                    # Make sure that the visible biases are in the same format as the weight matrix
                    tmp = zeros((1, len(visible_hidden_biases[i])))
                    tmp[0] = visible_hidden_biases[i]
                    self.weight_matrices_added_biases.append(append(weight_matrices[i].T, tmp, axis=0))
            if not os.path.exists(env_paths.get_dbn_batches_lst_path()):
                self.generate_large_batch()
        self.large_batches_lst = load_large_batches_lst()
        if self.binary_output:
            self.generate_sampled_noise(self.weight_matrices_added_biases)

    def run_finetuning(self, epochs):
        """
        Run the train and test error evaluation and the backpropagation using conjugate gradient to optimize the
        weights in order to make the DBN perform better.

        @param epochs: The number of epochs to run the finetuning for.
        """
        self.train_error = {}
        self.test_error = {}
        dbn.save_dbn(self.weight_matrices_added_biases, self.train_error, self.test_error, self.fout())
        for epoch in range(epochs):
            self.fout('Backprop: Epoch ' + str(epoch + 1))
            result_queue = Manager().Queue()
            w_queue = Manager().Queue()

            # Start backprop process
            proc = Process(target=self.backpropagation, args=(epoch, self.weight_matrices_added_biases, w_queue,))
            proc.start()
            # Start error eval processes
            evaluations = []
            evaluations.append((
                self.weight_matrices_added_biases, epoch, True, data_processing.get_batch_list(training=True),
                result_queue,
                self.binary_output))
            evaluations.append((
                self.weight_matrices_added_biases, epoch, False, data_processing.get_batch_list(training=False),
                result_queue, self.binary_output))
            p = Pool(cpu_count())
            p.map_async(error, evaluations)
            p.close()

            # Join multiple processes
            p.join()
            proc.join()
            self.weight_matrices_added_biases = w_queue.get()

            # Print and save error estimates
            for e in range(2):
                out = result_queue.get()
                if out[0]:
                    self.train_error[epoch] = out[2]
                    self.fout(out[1])
                else:
                    self.test_error[epoch] = out[2]
                    self.fout(out[1])

            # Save DBN
            dbn.save_dbn(self.weight_matrices_added_biases, self.train_error, self.test_error, self.fout())

    def generate_sampled_noise(self, weight_matrices_added_biases):
        for largebatch in self.large_batches_lst:
            x = load_large_batch(largebatch)
            gaussian_noise = random.normal(0, 4, (len(x),
                                                  weight_matrices_added_biases[
                                                      (len(weight_matrices_added_biases) / 2) - 1].shape[
                                                      1]))  # Gaussian noise matrix variance 16 and mean 0.
            self.sampled_noise.append(gaussian_noise)

    def generate_large_batch(self):
        """
        Generate larger batches to speed up the finetuning process.
        """
        large_batch_size = 1000
        large_batches_lst = []
        batches_split = []
        batches_tmp = []
        for batch in self.batches:
            batches_tmp.append(batch)
            if batch % large_batch_size == 0 and self.batches[-1] - batch >= large_batch_size:
                large_batches_lst.append(int(batch))
                batches_split.append(batches_tmp)
            elif batch == self.batches[-1]:
                large_batches_lst.append(int(batch))
                batches_split.append(batches_tmp)
        save_large_batches_lst(large_batches_lst)
        p = Pool(6)
        p.map_async(generate_large_batch_parallel, [(batches, large_batch_size) for batches in batches_split])
        p.close()
        p.join()


    def backpropagation(self, epoch, weight_matrices_added_biases, queue):
        """
        Run backpropagation for all large batches.

        @param weight_matrices_added_biases: The weight matrices added biases.
        @param queue: A multiprocessing queue where the weight matrices and biases should be added to.
        """
        batch_count = 0
        for largebatch in self.large_batches_lst:
            print 'Large batch: %d of %d' % (self.large_batches_lst.index(largebatch) + 1, len(self.large_batches_lst))

            x = load_large_batch(largebatch)
            if self.binary_output:
                self.current_sampled_noise = self.sampled_noise[batch_count]

            collected_weights = []
            weight_sizes = []
            for w in weight_matrices_added_biases:
                weight_sizes.append(len(w) - 1)
                collected_weights += reshape(w, (1, len(w) * len(w[0]))).tolist()[0]

            # Transform datatypes
            collected_weights = array(collected_weights)
            weight_sizes.append(weight_sizes[0])
            weight_sizes = array(weight_sizes)
            weights, _, _ = minimize(collected_weights, self.get_grad_and_error, (weight_sizes, x), maxnumlinesearch=3,
                                     verbose=True)
            weight_matrices_added_biases = self.__convert__(weights, weight_sizes)
            batch_count += 1
        queue.put(weight_matrices_added_biases)


    def get_grad_and_error(self, weights, weight_sizes, x):
        """
        Calculate the error function and the 
        gradient for the conjugate gradient method.

        @param weights: The weight matrices added biases in one single list.
        @param weight_sizes: The size of each of the weight matrices.
        @param x: The BOW.
        """
        weights = self.__convert__(weights, weight_sizes)
        x = append(x, ones((len(x), 1), dtype=float64), axis=1)
        if self.binary_output:
            xout, z_values = generate_output_data(x, weights, binary_output=self.binary_output,
                                                  sampled_noise=self.current_sampled_noise)
        else:
            xout, z_values = generate_output_data(x, weights, self.binary_output)
        x[:, :-1] = get_norm_x(x[:, :-1])
        f = -sum(x[:, :-1] * log(xout))  # Cross-entropy error function

        # Gradient
        number_of_weights = len(weights)
        gradients = []
        delta_k = None
        for i in range(number_of_weights - 1, -1, -1):
            if i == number_of_weights - 1:
                delta = (xout - x[:, :-1])
                grad = dot(z_values[i - 1].T, delta)
            elif i == (number_of_weights / 2) - 1:
                delta = dot(delta_k, weights[i + 1].T)
                # delta = dot(delta_k,weights[i+1].T)*z_values[i]*(1-z_values[i])
                delta = delta[:, :-1]
                grad = dot(z_values[i - 1].T, delta)
            elif i == 0:
                delta = dot(delta_k, weights[i + 1].T) * z_values[i] * (1 - z_values[i])
                delta = delta[:, :-1]
                grad = dot(x.T, delta)
            else:
                delta = dot(delta_k, weights[i + 1].T) * z_values[i] * (1 - z_values[i])
                delta = delta[:, :-1]
                grad = dot(z_values[i - 1].T, delta)
            delta_k = delta
            gradients.append(grad)

        gradients.reverse()
        gradients_formatted = []
        for g in gradients:
            gradients_formatted = append(gradients_formatted, reshape(g, (1, len(g) * len(g[0])))[0])
        return f, array(gradients_formatted)


    def __convert__(self, weights, dim):
        """
        Accept the weight matrices as one dimensional array and reshape to 2-dimensional matrices corresponding
        to the dimensions.

        @param weights: 1-dimensional array of weights.
        @param dim: list containing the dimensions of each weight matrix.
        """
        reshaped_weights = []

        position = 0
        for i in range(len(dim) - 1):
            reshaped_weights.append(
                reshape(weights[position:position + ((dim[i] + 1) * dim[i + 1])], ((dim[i] + 1), dim[i + 1])))
            position += (dim[i] + 1) * dim[i + 1]
        return reshaped_weights


def generate_output_data(x, weight_matrices_added_biases, binary_output=False, sampled_noise=None):
    """
    Compute forwards-pass in the deep autoencoder and compute the output.

    @param x: The BOW.
    @param weight_matrices_added_biases: The weight matrices added biases.
    @param binary_output: If the output of the DBN must be binary. If so, Gaussian noise will be added to bottleneck.
    @param sampled_noise: The gaussian noise matrix in case of binary output units.
    """
    z_values = []
    NN = sum(x, axis=1)
    for i in range(len(weight_matrices_added_biases) - 1):
        if i == 0:
            z = dbn.sigmoid(dot(x[:, :-1], weight_matrices_added_biases[i][:-1, :]) + outer(NN,
                                                                                            weight_matrices_added_biases[
                                                                                                i][-1, :]))
        elif i == (len(weight_matrices_added_biases) / 2) - 1:
            act = dot(z_values[i - 1], weight_matrices_added_biases[i])
            if binary_output and not sampled_noise is None:
                z = act + sampled_noise
            else:
                z = act
        else:
            z = dbn.sigmoid(dot(z_values[i - 1], weight_matrices_added_biases[i]))

        z = append(z, ones((len(x), 1), dtype=float64), axis=1)
        z_values.append(z)

    neg_vis = dot(z_values[-1], weight_matrices_added_biases[-1])
    softmax_value = dbn.softmax(neg_vis)
    xout = softmax_value
    if len(xout[xout == 0]) > 0:
        w = where(xout == 0)
        for i in range(len(w[0])):
            row = w[0][i]
            col = w[1][i]
            xout[row, col] = finfo(float).eps
    return xout, z_values


def error(args):
    """
    Compute the training or testing error on the unfolded network.
    """
    weights, epoch, training, batches, queue, binary_output = args
    err = 0
    for batch in range(len(batches)):
        x = get_bag_of_words_matrix(batches[batch]) if training else get_bag_of_words_matrix(batches[batch],
                                                                                             training=False)
        x = append(x, ones((len(x), 1)), axis=1)
        xout, _ = generate_output_data(x, weights, binary_output=binary_output)
        x[:, :-1] = get_norm_x(x[:, :-1])
        err -= sum(x[:, :-1] * log(xout))

    if training:
        out = 'Train error before epoch[' + str(epoch + 1) + ']: ' + str(err / (len(batches)))
    else:
        out = 'Test error before epoch[' + str(epoch + 1) + ']: ' + str(err / (len(batches)))

    queue.put([training, out, err / (len(batches))])


def save_large_batches_lst(lst):
    s.dump(lst, open(env_paths.get_dbn_batches_lst_path(), 'wb'))


def load_large_batches_lst():
    return s.load(open(env_paths.get_dbn_batches_lst_path(), 'rb'))


def save_large_batch(batch, data):
    s.dump(data.tolist(), open(env_paths.get_dbn_large_batch_data_path(batch), 'wb'))


def load_large_batch(batch):
    return array(s.load(open(env_paths.get_dbn_large_batch_data_path(batch), 'rb')))


def generate_large_batch_parallel(args):
    batches, large_batch_size = args
    large_batches_lst = []
    x = None
    for batch in batches:
        # Append input data.
        x_tmp = get_bag_of_words_matrix(batch)
        if x is None:
            x = x_tmp
        else:
            x = append(x, x_tmp, axis=0)

        if len(x) == large_batch_size and batches[-1] - batch >= large_batch_size:
            large_batches_lst.append(int(batch))
            save_large_batch(int(batch), x)
            x = None
        elif len(x) > large_batch_size and batch == batches[-1]:
            large_batches_lst.append(int(batch))
            save_large_batch(int(batch), x)
            x = None
        elif batch == batches[-1]:
            large_batches_lst.append(int(batch))
            save_large_batch(int(batch), x)
            x = None


def get_norm_x(x_matrix):
    """
    Normalize the BOW matrix and make sure not to do division by 0. Division by zero only occurs when there is no
    words in the document represented by the attributes vector, hence the BOW columns.
    @param x_matrix: The BOW matrix.
    @return: The normalized BOW.
    """
    sum_x = sum(x_matrix, axis=1)
    indices = where(sum_x == 0)
    for i in indices:
        sum_x[i] = 1.
    norm_x = x_matrix / sum_x[newaxis].T
    return norm_x
