import numpy as np
import net
import parser


path = "/Users/mortenflood/Documents/Advanced_Data_Mining/Project/YearPredictionMSD.txt"
path2 = "testset.txt"


def launch(file):
    print "Start"
    training_data, training_labels, test_data, test_labels = parser.load_dataset_decades_zero_index(file)
    # training_data, training_labels, test_data, test_labels = parser.load_dataset_zero_index(file)

    print "Dataset loaded"

    size = training_data.shape[0]
    x = training_data[range(size-10000)]
    y = training_labels[range(size-10000)]
    x_val = training_data[range(size-10000, size)]
    y_val = training_labels[range(size-10000, size)]

    print "Normalize"
    x = parser.mean_subtraction(x, x)
    x_val = parser.mean_subtraction(x_val, x)
    test_data = parser.mean_subtraction(test_data, x)

    # x = parser.normalize(x, x)
    # x_val = parser.normalize(x_val, x)
    # test_data = parser.normalize(test_data, x)

    print "Start training ..."

    net.run_training(x, y, x_val, y_val, test_data, test_labels)


launch(path)
