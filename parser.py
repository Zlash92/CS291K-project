import numpy as np



def load_dataset(file):
    data = np.loadtxt(file, delimiter=',')
    x = np.delete(data, np.s_[0], axis=1)
    y = np.delete(data, np.s_[1:], axis=1)

    # print data
    # print data.shape

    # dict = {}
    # for i in range(90000):
    #     if y[i, 0] not in dict.keys():
    #         dict[y[i][0]] = 1
    #     else:
    #         dict[y[i][0]] += 1
    # print dict
    #
    # dictTest = {}
    # for i in range(90000, 100000):
    #     if y[i, 0] not in dictTest.keys():
    #         dictTest[y[i][0]] = 1
    #     else:
    #         dictTest[y[i][0]] += 1
    #
    # print dictTest

    print y
    """"
    Total number of samples in dataset.txt: 515,345

    Info from dataset source:
    You should respect the following train / test split:
    train: first 463,715 examples
    test: last 51,630 examples
    It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
    """
    training_data = x[0:463715]
    training_labels = y[0:463715]
    test_data = x[463715:515344]
    test_labels = y[463715:515344]

    return training_data, training_labels, test_data, test_labels

def load_dataset_decade_mode(file):
    data = np.loadtxt(file, delimiter=',')
    x = np.delete(data, np.s_[0], axis=1)
    y = np.delete(data, np.s_[1:], axis=1)

    decade_y = []
    pass

load_dataset('testset.txt')
# load_dataset('dataset.txt')