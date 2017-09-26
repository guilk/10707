import os
import numpy as np

expit = lambda x: 1.0/(1+np.exp(-x))

def load_data(db_path):
    raw_data = np.loadtxt(fname=db_path, dtype=np.float32, delimiter=',')
    data = raw_data[:, :-1]
    label = raw_data[:, -1].astype(np.int)
    return data,label

def shuffle(data, label):
    num_data = data.shape[0]
    p = np.random.permutation(num_data)
    return data[p], label[p]

def init_fc(n_in, n_out, act_type='sigmoid'):
    if act_type == 'sigmoid':
        low = -4 * np.sqrt(6. / (n_in + n_out))
        high = 4 * np.sqrt(6. / (n_in + n_out))
    else:
        low = -np.sqrt(6. / (n_in + n_out))
        high = np.sqrt(6. / (n_in + n_out))

    W1 = np.asarray(np.random.uniform(low, high, size=(n_in, n_out)), dtype=float)
    b1 = np.zeros((n_out,), dtype=float)
    return W1,b1

def sigmoid(x):
    # Prevent overflow
    x = np.clip(x, -500, 500)

    # Calculaet activation signal
    x = expit(x)

    return x

def sigmoid_deri(x):
    # Prevent overflow
    x = np.clip(x, -500, 500)

    # Calculaet activation signal
    x = expit(x)

    derivate = np.multiply(x, 1-x)
    return derivate

def ReLU(x):
    return np.maximum(0, x)

def ReLU_deri(x):
    return (x>0).astype(float)


def tanh(x):
    x = np.tanh(x)
    return x

def tanh_deri(x):
    x = np.tanh(x)
    return 1-np.power(x,2)


