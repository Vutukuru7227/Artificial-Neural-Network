import sys
import random
import math
import collections
import numpy as np


class NeuralNet:

    def __init__(self):
        pass

    def assign_weights(self, number_of_neurons_in_each_layer, no_of_feature_values):
        neural_network_weight_list = collections.defaultdict(list)
        j = 1

        for each_hidden_layer in number_of_neurons_in_each_layer:
            neural_network_weight_list[j].append(np.random.uniform(-1, 1, (each_hidden_layer, no_of_feature_values)))
            no_of_feature_values = each_hidden_layer + 1
            j = j+1
        neural_network_weight_list[j].append(np.random.uniform(-1, 1, (1, no_of_feature_values)))
        # print("================Weights====================")
        # print(neural_network_weight_list)
        return neural_network_weight_list

    @staticmethod
    def sigmoid(x):
        return 1/(1+math.exp(-x))

    def forward_pass(self, neural_network_weight_list, each_instance):
        result_matrix = collections.defaultdict(list)

        for i in neural_network_weight_list.keys():
            result_matrix[i].extend(np.matmul(each_instance, np.transpose(neural_network_weight_list[i][0])))
            result_sigmoid_matrix = list()
            for j in result_matrix[i]:
                j = NeuralNet.sigmoid(j)
                result_sigmoid_matrix.append(j)
            result_matrix[i] = result_sigmoid_matrix
            each_instance = [1] + result_matrix[i]
        return result_matrix

    def backward_pass(self, learning_rate, output, target_value, output_network, neural_net_weights_list, each_instance):
        rate_change_matrix = collections.defaultdict(list)
        neuron_number = len(neural_net_weights_list.keys())

        while neuron_number >= 1:

            if neuron_number == len(neural_net_weights_list.keys()):
                rate_change_matrix = self.backward_pass_output_neurons(output,target_value, neural_net_weights_list)
                rate_change_matrix.get(neuron_number)[0] = rate_change_matrix.get(neuron_number)[0][0];
                neuron_number = neuron_number - 1
            else:
                rate_change_matrix = self.backward_pass_hidden_layer_neurons(neural_net_weights_list, output_network,rate_change_matrix)
                neuron_number = neuron_number - 1

        i = len(output_network.keys())-1
        while i > 0:
            output_network[i-1] = [1] + output_network[i-1]

            if i == 1:
                neural_net_weights_list[i][0] += np.outer(rate_change_matrix[i][0], each_instance) * learning_rate
                i = i - 1
            else:
                neural_net_weights_list[i][0] += np.outer(rate_change_matrix[i][0], output_network[i-1]) * learning_rate
                i = i - 1

        return neural_net_weights_list

    def backward_pass_output_neurons(self, output, target_value, neural_network_weight_list):
        rate_change_matrix = collections.defaultdict(list)
        rc = list()
        rc.append((target_value - output) * (1 - output) * output)
        rate_change_matrix[len(neural_network_weight_list.keys())].append(rc)
        return rate_change_matrix

    def backward_pass_hidden_layer_neurons(self, neural_network_weight_list, output_network, rate_change):

        i = len(neural_network_weight_list.keys())

        while i >= 1:
            value = np.matmul(np.transpose(rate_change.get(i)[0]), neural_network_weight_list.get(i)[0])
            j = len(output_network[i-1]) - 1
            value = value[1:]
            while j >= 0:
                value[j] = output_network[i-1][j] * (1-output_network[i-1][j]) * value[j]
                j = j - 1
            rate_change[i-1].append(value)
            i = i - 1

        return rate_change


def read_data_set(input_data_set):
    input_layer = np.loadtxt(input_data_set, delimiter=',', dtype=np.float64)
    input_layer = np.around(input_layer, decimals=2)

    return input_layer


def create_x_list(input_layer):
    x_list = [1]*len(input_layer)
    input_layer = np.column_stack((x_list, input_layer))

    return input_layer


def print_neurons(neural_network_weight_list):

    for i in neural_network_weight_list.keys():
        if i != len(neural_network_weight_list):
            print("Hidden Layer " + str(i))
        else:
            print("Output Layer:")
        counter = 1
        for k in neural_network_weight_list[i][0]:
            print("Neuron" + str(counter) + " weights:  " + '  '.join(str(weight) for weight in k))
            counter += 1


def main():
    # TODO: Reading the inputs from terminal
    input_data_set = sys.argv[1]
    training_percentage = (sys.argv[2])
    maximum_iterations = (sys.argv[3])
    number_of_hidden_layers = (sys.argv[4])

    number_of_neurons_in_each_hidden_layer = list()

    for i in range(int(number_of_hidden_layers)):
        number_of_neurons_in_each_hidden_layer.append(int(sys.argv[5+i]))

    # print(input_data_set)
    # print(training_percentage)
    # print(maximum_iterations)
    # print(number_of_hidden_layers)
    # print(number_of_neurons_in_each_hidden_layer)

    input_layer = read_data_set(input_data_set)
    input_layer = create_x_list(input_layer)
    #print(input_layer)


    # Shuffling the input data set
    #print("===========Shuffled==========")
    input = random.sample(range(0, len(input_layer)), len(input_layer))
    #print(input)

    # TODO: Data Splitting
    m = 0
    training_set = list()
    testing_set = list()
    for i in input:
        if m < int((len(input_layer) * (float(training_percentage)/100))):
            training_set.append(input_layer[i])
        else:
            testing_set.append(input_layer[i])
        m = m + 1

    training_set = np.array(training_set)
    testing_set = np.array(testing_set)

    training_set_feature_values = training_set[:, :len(training_set[0])-1]
    training_set_class_label = training_set[:, len(training_set[0])-1:]

    # print("===============Training Set Feature Values================")
    # print(training_set_feature_values)
    #
    # print("===============Training Set Class Labels================")
    # print(training_set_class_label)

    testing_set_feature_values = testing_set[:, :len(testing_set[0])-1]
    testing_set_class_label = testing_set[:, len(testing_set[0])-1:]

    # print("===============Testing Set Feature Values================")
    # print(testing_set_feature_values)
    #
    # print("===============Testing Set Class Labels================")
    # print(testing_set_class_label)

    neural_network_instance = NeuralNet()
    neural_network_weight_list = neural_network_instance.assign_weights(number_of_neurons_in_each_hidden_layer, len(training_set_feature_values[0]))

    error = 999999
    iterations_limit = int(maximum_iterations)
    learning_rate = 0.5
    i = 0
    while error > float(maximum_iterations) and iterations_limit > 0:
        error = 0
        instance = 0
        for each_instance in training_set_feature_values:
            output_network = neural_network_instance.forward_pass(neural_network_weight_list, each_instance)
            output = output_network[len(neural_network_weight_list.keys())][0]
            neural_network_weight_list = neural_network_instance.backward_pass(learning_rate, output, training_set_class_label[instance], output_network, neural_network_weight_list, each_instance)
            error = error + math.pow(training_set_class_label[instance][0] - output, 2)
            instance = instance + 1
        error = error / (2 * int((len(input_layer) * (float(training_percentage)/100))))
        iterations_limit = iterations_limit - 1
    training_error = error

    error = 0
    instance = 0
    for each_instance in testing_set_feature_values:
        output_network_1 = neural_network_instance.forward_pass(neural_network_weight_list, each_instance)
        output = output_network_1[len(neural_network_weight_list.keys())][0]
        error += pow((testing_set_class_label[instance][0] - output), 2)
        instance += 1
    error = error / (2 * int((len(input_layer) * (float(100 - int(training_percentage)) / 100))))
    testing_error = error

    print_neurons(neural_network_weight_list)

    print("Training Error", training_error)
    print("Testing Error", testing_error)

if __name__ == '__main__':
    main()