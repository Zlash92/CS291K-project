import numpy as np
import net


def load_dataset(file):
    # f = open(file)
    # data = f.readlines()
    # print data
    # print "Start"
    # data = np.loadtxt(file, delimiter=',')
    # x = np.delete(data, np.s_[0], axis=1)
    # y = np.delete(data, np.s_[1:], axis=1)
    # "reshape y"
    # new_y = np.zeros((y.shape[0],), dtype=np.int)
    # for i in range(len(y)):
    #     new_y[i] = int(y[i, 0]-1922)
    #
    # print "GO"
    x = np.load("X_train.npy")
    y = np.load("Y_train.npy")

    np.save("X_train", x)
    np.save("Y_train", y)

    net.run_training(x, y)

load_dataset('YearPredictionMSD.txt')