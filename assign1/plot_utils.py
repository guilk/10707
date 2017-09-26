import matplotlib.pyplot as plt
import numpy as np


def plot_loss(tr_loss, val_loss):
    plt.figure()
    mean_tr_loss = np.mean(tr_loss, axis=0).squeeze()
    mean_val_loss = np.mean(val_loss, axis=0).squeeze()
    X = np.asarray(range(200))

    plt.plot(X, mean_tr_loss, color='r', label='train_loss')
    plt.plot(X, mean_val_loss, color='b', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()

    plt.axis([0, 200, 0, 1])
    # plt.legend(label='')
    # plt.show()
    plt.savefig('../figs/6a.eps', format='eps', dpi=100)
    plt.close()

def plot_acc(tr_acc, val_acc):
    plt.figure()
    mean_tr_acc = np.mean(tr_acc, axis=0).squeeze()
    mean_val_acc = np.mean(val_acc, axis=0).squeeze()
    X = np.asarray(range(200))

    plt.plot(X, 1-mean_tr_acc, color='r', label='train_acc')
    plt.plot(X, 1-mean_val_acc, color='b', label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.legend()

    plt.axis([0, 200, 0, 0.7])
    # plt.legend(label='')
    # plt.show()
    plt.savefig('../figs/6b.eps', format='eps', dpi=100)
    plt.close()

def plot_lr(train_loss, val_loss, train_acc, val_acc, lr_rates):
    plt.figure()
    X = np.asarray(range(200))
    # plt.subplot(211)
    for index, lr_rate in enumerate(lr_rates):
        plt.plot(X, train_loss[index].squeeze(), label='Trn Loss lr {}'.format(lr_rate))
        plt.plot(X, val_loss[index].squeeze(), label='Val Loss lr {}'.format(lr_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.savefig('../figs/6d-loss.eps', format='eps', dpi=100)
    plt.close()

    plt.figure()
    for index, lr_rate in enumerate(lr_rates):
        plt.plot(X, 1-train_acc[index].squeeze(), label='Trn Acc lr {}'.format(lr_rate))
        plt.plot(X, 1-val_acc[index].squeeze(), label='Val Acc lr {}'.format(lr_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.axis([0, 200, 0, 0.7])
    plt.legend()
    plt.savefig('../figs/6d-error.eps', format='eps', dpi=100)
    plt.close()

def plot_momentum(train_loss, val_loss, train_acc, val_acc, momentums):
    plt.figure()
    X = np.asarray(range(200))
    # plt.subplot(211)
    for index, lr_rate in enumerate(momentums):
        plt.plot(X, train_loss[index].squeeze(), label='Trn Loss lr {}'.format(lr_rate))
        plt.plot(X, val_loss[index].squeeze(), label='Val Loss lr {}'.format(lr_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.savefig('../figs/6d-momentum-loss.eps', format='eps', dpi=100)
    plt.close()

    plt.figure()
    for index, lr_rate in enumerate(momentums):
        plt.plot(X, 1-train_acc[index].squeeze(), label='Trn Acc momentum {}'.format(lr_rate))
        plt.plot(X, 1-val_acc[index].squeeze(), label='Val Acc momentum {}'.format(lr_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.axis([0, 200, 0, 0.7])
    plt.legend()
    plt.savefig('../figs/6d-momentum-error.eps', format='eps', dpi=100)
    plt.close()

def plot_hidden(train_loss, val_loss, train_acc, val_acc, num_hiddens):
    plt.figure()
    X = np.asarray(range(200))
    # plt.subplot(211)
    for index, lr_rate in enumerate(num_hiddens):
        plt.plot(X, train_loss[index].squeeze(), label='Trn Loss hidden {}'.format(lr_rate))
        plt.plot(X, val_loss[index].squeeze(), label='Val Loss hidden {}'.format(lr_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.savefig('../figs/6f-loss.eps', format='eps', dpi=100)
    plt.close()

    plt.figure()
    for index, lr_rate in enumerate(num_hiddens):
        plt.plot(X, 1-train_acc[index].squeeze(), label='Trn Acc hidden {}'.format(lr_rate))
        plt.plot(X, 1-val_acc[index].squeeze(), label='Val Acc hidden {}'.format(lr_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.axis([0, 200, 0, 0.7])
    plt.legend()
    plt.savefig('../figs/6f-error.eps', format='eps', dpi=100)
    plt.close()





