import sys
import random
import numpy as np

class Neuron:

    def __init__(self):
        pass

class NeuralNet:

    def __init__(self):
        pass

    def forward_pass(self):
        pass


def read_data_set(input_data_set):
    input_layer = np.loadtxt(input_data_set, delimiter=',', dtype=np.float64)
    input_layer = np.around(input_layer, decimals=2)

    return input_layer


def create_x_list(input_layer):
    x_list = [1]*len(input_layer)
    x_list = np.column_stack((x_list, input_layer))

    return x_list


def main():
    # Reading the inputs from terminal
    input_data_set = sys.argv[1]
    training_percentage = float(sys.argv[2])
    maximum_iterations = int(sys.argv[3])
    number_of_hidden_layers = int(sys.argv[4])

    number_of_neurons_in_each_hidden_layer = []

    for i in range(number_of_hidden_layers):
        number_of_neurons_in_each_hidden_layer.append(int(sys.argv[5+i]))

    # print(input_data_set)
    # print(training_percentage)
    # print(maximum_iterations)
    # print(number_of_hidden_layers)
    # print(number_of_neurons_in_each_hidden_layer)

    input_layer = read_data_set(input_data_set)
    input_layer = create_x_list(input_layer)
    print(input_layer)


    #Shuffling the input data set
    print("===========Shuffled==========")
    random.shuffle(input_layer)
    print(input_layer)


    #TODO: Data Splitting
    training_set = list()

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


if __name__ == '__main__':
    main()






