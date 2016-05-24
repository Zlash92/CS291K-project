import tensorflow as tf
import numpy as np


def run_training(x_images, y_labels, x_val, y_val, x_test, y_test):

    x = tf.placeholder(tf.float32, shape=[None, 90])
    y = tf.placeholder(tf.int32, shape=[None, ])  # TODO: ??
    keep_hidden = tf.placeholder("float")

    neurons = 512
    lr = 1e-3
    batch_size = 100
    epoch = int(round(x_images.shape[0]/batch_size))

    logits = inference(x, keep_hidden, neurons)
    loss_ = loss(logits, y)
    train = training(loss_, lr)
    accuracy = evaluation(logits, y)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for steps in range(100000):

        rand = np.random.randint(0, x_images.shape[0], batch_size)
        x_batch = x_images[rand]
        y_batch = y_labels[rand]

        feed_dict = {x: x_batch, y: y_batch, keep_hidden: 0.5}

        _, loss_val = sess.run([train, loss_], feed_dict=feed_dict)

        if steps % 200 == 0:
            corr = sess.run(accuracy, feed_dict={x: x_val, y:y_val, keep_hidden: 1.0})
            acc = float(corr)/y_val.shape[0] * 100.0
            print "Step ", steps, "  - Validation accuracy: ", acc
            print "Training loss: ", loss_val
        if steps % epoch == 0:
            corr = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_hidden: 1.0})
            acc = float(corr) / y_test.shape[0] * 100.0
            print "Epoch ", int(steps/epoch)," Step ", steps, "  - Test accuracy: ", acc


def evaluate():
    pass


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def inference(x, keep_hidden, neurons):
    output_nerons = 9
    # Fully connected layer 1
    epsilon = 1e-3
    with tf.name_scope('hidden1'):
        W1 = weight_variable([90, neurons])
        b1 = bias_variable([])
        z1 = tf.matmul(x, W1) + b1

        mean1, var1 = tf.nn.moments(z1, [0])
        scale1 = tf.Variable(tf.ones([neurons]))
        beta1 = tf.Variable(tf.zeros([neurons]))
        z1_bn = tf.nn.batch_normalization(z1, mean1, var1, beta1, scale1, epsilon)

        activation1 = tf.nn.relu(z1_bn)
        activation1 = tf.nn.dropout(activation1, keep_hidden)

        # activation1_l2_norm = tf.nn.l2_normalize(activation1, dim=0)

    # with tf.name_scope('hidden2'):
    #     W2 = weight_variable([neurons, neurons])
    #     b2 = bias_variable([])
    #     z2 = tf.matmul(activation1, W2) + b2
    #
    #     mean2, var2 = tf.nn.moments(z2, [0])
    #     scale2 = tf.Variable(tf.ones([neurons]))
    #     beta2 = tf.Variable(tf.zeros([neurons]))
    #     z2_bn = tf.nn.batch_normalization(z2, mean2, var2, beta2, scale2, epsilon)
    #
    #     activation2 = tf.nn.relu(z2_bn)
    #     activation2 = tf.nn.dropout(activation2, keep_hidden)


    # # #Output layer
    with tf.name_scope('softmax_layer'):
        W_out = weight_variable([neurons, output_nerons])
        b_out = bias_variable([output_nerons])
        #softmax = tf.nn.softmax(tf.matmul(activation2, weights) + bias)
        logits = tf.matmul(activation1, W_out) + b_out

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss_value = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss_value


def training(loss_value, learning_rate):
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss_value)
    #train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value)
    return train


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_sum(tf.cast(correct, tf.int32)) # reduce_sum instead?
    return accuracy
