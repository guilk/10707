import os
import numpy as np
from data_iterator import DataIterator
from utils import *
import matplotlib.pyplot as plt
from plot_utils import *
import tensorlayer as tl

def calculate_loss(W1, b1, W2, b2, reg_lambda, X, y):
    num_data = X.shape[0]
    z1 = X.dot(W1) + b1
    # a1 = tanh(z1)
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_data),y])
    data_loss = np.sum(correct_logprobs)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_data * data_loss

def predict(W1, b1, W2, b2, X, y):
    z1 = X.dot(W1) + b1
    # a1 = tanh(z1)
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores/ np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

if __name__ == '__main__':
    '''
    Implement 784->sigmod->100->softmax->10 single layer neural network
    '''

    # Load data
    tr_data_path = '../data/digitstrain.txt'
    tr_data, tr_label = load_data(tr_data_path)

    val_data_path = '../data/digitsvalid.txt'
    val_data, val_label = load_data(val_data_path)
    #
    # ts_data_path = '../data/digitstest.txt'
    # ts_data, ts_label = load_data(ts_data_path)

    epochs = 200
    batch_size = 32
    reg_lambda = 0.0001
    lr_rate = 0.01
    mf = 0.5
    # lr_rates = [0.01, 0.1, 0.2, 0.5]
    # mfs = [0.0, 0.5, 0.9]
    num_hiddens = [20, 100, 200, 500]
    num_data = tr_data.shape[0]
    num_batches = num_data/batch_size
    act_type = 'sigmoid'
    num_repeat = 1

    # train_loss_total = np.zeros((num_repeat, epochs))
    # val_loss_total = np.zeros((num_repeat, epochs))
    # tr_acc_total = np.zeros((num_repeat, epochs))
    # val_acc_total = np.zeros((num_repeat, epochs))

    # train_loss_total = np.zeros((len(lr_rates), epochs))
    # val_loss_total = np.zeros((len(lr_rates), epochs))
    # tr_acc_total = np.zeros((len(lr_rates), epochs))
    # val_acc_total = np.zeros((len(lr_rates), epochs))

    # train_loss_total = np.zeros((len(mfs), epochs))
    # val_loss_total = np.zeros((len(mfs), epochs))
    # tr_acc_total = np.zeros((len(mfs), epochs))
    # val_acc_total = np.zeros((len(mfs), epochs))
    train_loss_total = np.zeros((len(num_hiddens), epochs))
    val_loss_total = np.zeros((len(num_hiddens), epochs))
    tr_acc_total = np.zeros((len(num_hiddens), epochs))
    val_acc_total = np.zeros((len(num_hiddens), epochs))

    # for index_repeat in range(num_repeat):
    # for index_mf, mf in enumerate(mfs):
    for index_hidden, num_hidden in enumerate(num_hiddens):
        # Intialize the first fully-connected layer
        W1,b1 = init_fc(n_in = 784, n_out = num_hidden, act_type='sigmoid')
        W2,b2 = init_fc(n_in = num_hidden, n_out = 10, act_type='sigmoid')

        mf_W1 = np.zeros_like(W1)
        mf_b1 = np.zeros_like(b1)
        mf_W2 = np.zeros_like(W2)
        mf_b2 = np.zeros_like(b2)

        for index_epoch in range(epochs):

            tr_data, tr_label = shuffle(tr_data, tr_label)
            for index_batch in range(num_batches):
                X = tr_data[index_batch*batch_size:(index_batch+1)*batch_size,:]
                y = tr_label[index_batch*batch_size:(index_batch+1)*batch_size]
                # Foward pass
                z1 = X.dot(W1) + b1
                # a1 = tanh(z1) # tanh activation function
                a1 = sigmoid(z1)
                z2 = a1.dot(W2) + b2
                exp_scores = np.exp(z2)
                probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

                # Backward pass
                delta3 = probs
                delta3[range(batch_size),y] -= 1
                dW2 = (a1.T).dot(delta3)
                db2 = np.sum(delta3, axis=0)
                # delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) # tanh activation function
                delta2 = delta3.dot(W2.T) * (np.multiply(a1, 1-a1)) # sigmoid activation function
                dW1 = np.dot(X.T, delta2)
                db1 = np.sum(delta2, axis=0)


                # Add regularization terms
                dW2 += reg_lambda * W2
                dW1 += reg_lambda * W1

                # Gradient descent parameter update
                mf_W2 = mf * mf_W2 + lr_rate * dW2
                mf_b2 = mf * mf_b2 + lr_rate * db2
                mf_W1 = mf * mf_W1 + lr_rate * dW1
                mf_b1 = mf * mf_b1 + lr_rate * db1

                W1 -= mf_W1
                b1 -= mf_b1
                W2 -= mf_W2
                b2 -= mf_b2


            tr_loss = calculate_loss(W1, b1, W2, b2, reg_lambda, tr_data, tr_label)
            val_loss = calculate_loss(W1, b1, W2, b2, reg_lambda, val_data, val_label)
            train_loss_total[index_hidden, index_epoch] = tr_loss
            val_loss_total[index_hidden, index_epoch] = val_loss

            pred_tr_labels = predict(W1, b1, W2, b2, tr_data, tr_label)
            pred_val_labels = predict(W1, b1, W2, b2, val_data, val_label)
            tr_acc = 1.0 * np.sum(pred_tr_labels == tr_label) / tr_label.shape[0]
            val_acc = 1.0 * np.sum(pred_val_labels == val_label) / val_label.shape[0]
            tr_acc_total[index_hidden, index_epoch] = tr_acc
            val_acc_total[index_hidden, index_epoch] = val_acc
            print '{}th epoch, cross entropy loss on training is {}, ' \
                  'on val is {}'.format(index_epoch, tr_loss, val_loss)
    plot_hidden(train_loss_total, val_loss_total, tr_acc_total, val_acc_total, num_hiddens)


            #
            # pred_tr_labels = predict(W1, b1, W2, b2, tr_data, tr_label)
            # pred_val_labels = predict(W1, b1, W2, b2, val_data, val_label)
            # tr_acc = 1.0 * np.sum(pred_tr_labels == tr_label) / tr_label.shape[0]
            # val_acc = 1.0 * np.sum(pred_val_labels == val_label) / val_label.shape[0]
            # tr_acc_total[index_repeat, index_epoch] = tr_acc
            # val_acc_total[index_repeat, index_epoch] = val_acc

            # print '{}th epoch, validation accuracy is {}'.format(index_epoch, accuracy)

        # tl.visualize.W(W1, second=10, saveable=True, name='6c', fig_idx=2012)
    # mean_tr_loss = np.mean(train_loss_total, axis=0)
    # mean_val_loss = np.mean(val_loss_total, axis=0)
    # plot_acc(tr_acc_total, val_acc_total)
    # plot_loss(train_loss_total, val_loss_total)