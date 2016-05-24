import numpy as np

""""
Total number of samples in dataset.txt: 515,345

Info from dataset source:
You should respect the following train / test split:
train: first 463,715 examples
test: last 51,630 examples
It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
"""

# TODO: Change label types to int?
def load_dataset(file):
    data = np.loadtxt(file, delimiter=',')
    x = np.delete(data, np.s_[0], axis=1)
    y = np.delete(data, np.s_[1:], axis=1)
    y = y.flatten()

    training_data = x[0:463715]
    test_data = x[463715:x.shape[0]]

    training_labels = y[0:463715]
    test_labels = y[463715:x.shape[0]]

    return training_data, training_labels, test_data, test_labels

"""Labels ranged from 0-89 corresponding to 1922-2011, endpoints inclusive"""
def load_dataset_zero_index(file):
    data = np.loadtxt(file, delimiter=',')
    x = np.delete(data, np.s_[0], axis=1)
    y = np.delete(data, np.s_[1:], axis=1)
    y = y.astype(int)

    start_year = 1922

    y_mod = []
    for l in y:
        y_mod.append(l - start_year)

    y_mod = np.asarray(y_mod)
    # print y_mod

    training_data = x[0:463715]
    test_data = x[463715:515344]

    training_labels = y_mod[0:463715]
    test_labels = y_mod[463715:515344]

    return training_data, training_labels, test_data, test_labels


"""Labels ranged from 0-9 corresponding to 1920s - 2010s"""
def load_dataset_decades_zero_index(file):
    training_data, training_labels_init, test_data, test_labels_init = load_dataset(file)

    training_labels = reshape_to_decades(training_labels_init)
    test_labels = reshape_to_decades(test_labels_init)

    return training_data, training_labels, test_data, test_labels


def reshape_to_decades(y):
    result = []

    # 192 is 1922 stripped of last digit. Used for classification value
    for l in y:
        if l > 2009:
            result.append(8)
        else:
            temp_string = str(l)
            temp_string = temp_string[:3]
            value = int(temp_string)
            result.append(value-192)

    result = np.asarray(result)
    return result


def simple_load(file):
    data = np.loadtxt(file, delimiter=',')
    x = np.delete(data, np.s_[0], axis=1)
    y = np.delete(data, np.s_[1:], axis=1)

    return x, y


def mean_subtraction(x, xtr):
    x -= np.mean(xtr, axis=0)
    return x


def normalize(x, xtr):
    x /= np.std(xtr, axis=0)
    return x


# xt, yt = simple_load('testset.txt')
# yt = yt.flatten()
# print yt
# print reshape_to_decades(yt)


