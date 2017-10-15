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
            number_of_neurons_in_each_layer = each_hidden_layer + 1
            j = j+1
        neural_network_weight_list[j].append(np.random.uniform(-1, 1, (1, no_of_feature_values)))
        print("================Weights====================")
        print(neural_network_weight_list)
        return neural_network_weight_list

    @staticmethod
    def sigmoid(x):
        return 1/(1+math.exp(-x))

    def forward_pass(self, neural_network_weight_list, each_instance):
        result_matrix = collections.defaultdict(list)
        for i in neural_network_weight_list.keys():
            result_matrix[i].extend(np.dot(each_instance, np.transpose(neural_network_weight_list[i][0])))
            result_sigmoid_matrix = []
            for j in result_matrix[i]:
                result_sigmoid_matrix.append(NeuralNet.sigmoid(j))
        result_matrix[i] = result_sigmoid_matrix
        each_instance = [1] + result_matrix[i]
        return result_matrix

    def backward_pass_output_neurons(self, target_value, output, neural_network_weight_list):
        rate_change_matrix = collections.defaultdict(list)
        weight_change = (target_value - output) * output * (1 - output)
        rate_changed_value = [].append(weight_change)
        rate_change_matrix[len(neural_network_weight_list.keys())].append(rate_changed_value)
        return rate_change_matrix

    def backward_pass_hidden_layer_neurons(self, rate_change_matrix):
        pass


    def backward_pass(self, learning_rate, output, target_value, output_network, neural_net_weights_list, each_instance):
        rate_change_matrix = collections.defaultdict(list)
        neuron_number = len(neural_net_weights_list.keys())

        while neuron_number >= 1:

            if neuron_number == len(neural_net_weights_list.keys()):
                rate_change_matrix = self.backward_pass_output_neurons(target_value, output, neural_net_weights_list)
                rate_change_matrix.get(neuron_number)[0] = rate_change_matrix.get(neuron_number)[0][0];
                neuron_number = neuron_number - 1
            else:
                rate_change_matrix = self.backward_pass_hidden_layer_neurons(rate_change_matrix)
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



def read_data_set(input_data_set):
    input_layer = np.loadtxt(input_data_set, delimiter=',', dtype=np.float64)
    input_layer = np.around(input_layer, decimals=2)

    return input_layer


def create_x_list(input_layer):
    x_list = [1]*len(input_layer)
    x_list = np.column_stack((x_list, input_layer))

    return x_list


def main():
    # TODO: Reading the inputs from terminal

    input_data_set = sys.argv[1]
    training_percentage = float(sys.argv[2])
    maximum_iterations = int(sys.argv[3])
    number_of_hidden_layers = int(sys.argv[4])

    number_of_neurons_in_each_hidden_layer = []

    for i in range(number_of_hidden_layers):
        number_of_neurons_in_each_hidden_layer.append(int(sys.argv[5+i]))

    error = 101
    iterations = 0
    learning_rate = 0.5

    # print(input_data_set)
    # print(training_percentage)
    # print(maximum_iterations)
    # print(number_of_hidden_layers)
    # print(number_of_neurons_in_each_hidden_layer)

    input_layer = read_data_set(input_data_set)
    input_layer = create_x_list(input_layer)
    print(input_layer)


    # Shuffling the input data set
    print("===========Shuffled==========")
    random.shuffle(input_layer)
    print(input_layer)

    # TODO: Data Splitting

    print("No of total Instances:", len(input_layer))

    no_of_training_instances = int(len(input_layer) * (training_percentage/100))
    print("No of training set instances:", no_of_training_instances)

    no_of_testing_instances = int(len(input_layer) - no_of_training_instances)
    print("No of testing set instances:", no_of_testing_instances)

    training_set = input_layer[:no_of_training_instances]
    print("=================Training Set====================")
    print(training_set)

    testing_set = input_layer[no_of_training_instances:]
    print("=================Testing Set=====================")
    print(testing_set)

    training_set = np.array(training_set)
    testing_set = np.array(testing_set)

    training_set_feature_values = training_set[:, :- 1]
    training_set_class_label = training_set[:, -1:]

    print("===============Training Set Feature Values================")
    print(training_set_feature_values)

    print("===============Training Set Class Labels================")
    print(training_set_class_label)

    testing_set_feature_values = testing_set[:, :-1]
    testing_set_class_label = testing_set[:, -1:]

    print("===============Testing Set Feature Values================")
    print(testing_set_feature_values)

    print("===============Testing Set Class Labels================")
    print(testing_set_class_label)

    neural_network_instance = NeuralNet()
    neural_network_weight_list = neural_network_instance.assign_weights(number_of_neurons_in_each_hidden_layer, len(training_set_feature_values[0]))

    i = 0
    while iterations < maximum_iterations or error <= 0.05:
        error = 0
        instance = 0
        for each_instance in training_set_feature_values:
            output_network = neural_network_instance.forward_pass(neural_network_weight_list, each_instance)
            output = output_network[len(neural_network_weight_list.keys())][0]
            final_network = neural_network_instance.backward_pass(learning_rate, output, training_set_class_label[instance], output_network, neural_network_weight_list, each_instance)


        iterations = iterations + 1

if __name__ == '__main__':
    main()






