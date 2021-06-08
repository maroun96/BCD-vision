import time
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from torchvision.datasets import MNIST, FashionMNIST

from optimizers import CD
from datasets import load_dataset

import matplotlib.pyplot as plt

"""
This methods computes the accuracy of the model on test dataset
args:
    - model : the model to test
    - test_images
    - test_targets : the test labels
    - batch_size
"""
def compute_acc(model, test_images, test_targets, batch_size=100):
    model.eval()
    nb_right = 0
    with torch.no_grad():
        for b in range(0, test_images.size(0), batch_size):
            output = model(test_images.narrow(0, b, batch_size))
            predicted_class = torch.argmax(output, dim=1)
            for k in range(batch_size):
                if test_targets[b + k] == predicted_class[k]:
                    nb_right += 1
    return nb_right

"""
This methods trains the model on all epochs. It returns the final model, an historic if test accuracies and of train losses
args:
    - model : the model to train
    - optimizer : the optimizer choosen to train the model
    - train_images
    - train_targets : the training labels
    - test_images : 
    - test_targets : the test labels
    - batch_size
    - epochs : the number of epochs we want to train our model
"""
def train_model(model, optimizer, train_images, train_targets, test_images, test_targets, batch_size, epochs):
    criterion = nn.CrossEntropyLoss()
    model.train()
    prev_sum = 0
    train_loss = []
    accs = []
    for e in range(1, epochs + 1):
        sum_loss = 0
        nbr_batch = 0
        print("Epochs : {:d}/{:d}".format(e, epochs))
        for b in tqdm(range(0, train_images.size(0), batch_size)):
            output = model(train_images.narrow(0, b, batch_size))
            loss = criterion(output, train_targets.narrow(0, b, batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss += loss.detach().item()
            optimizer.step()
            nbr_batch += 1

        print("Training loss: {:.4f}".format(sum_loss))
        accs.append((compute_acc(model, test_images, test_targets, batch_size=100) / len(test_images))*100)
        print("Accuracy : {:.2f}".format(accs[-1]))

        #If the training does not converge, we divide the learning rate by 2
        if prev_sum < sum_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
        train_loss.append(sum_loss/nbr_batch)
        prev_sum = sum_loss

    return model, accs, train_loss

"""
Plots the results as a function of training loss against the epoch number
args:
    - dataset : the dataset on which we trained the model
    - n_block : the number of parameters to update if we used CD
    - train_losses : a dic which associates to the batch size an historic of training losses
    - n_rows : number of rows of the plot
    - ncols : number of cols of the plot
    - index : the index number of the plot
"""
def plot(dataset, n_block, train_losses, n_rows=1, ncols=1, index=1):
    x_axis = np.arange(len(list(train_losses.values())[0])) + 1

    plt.subplot(n_rows, ncols, index)
    for key, loss in train_losses.items():
        plt.plot(x_axis, loss, label="Batch size = {:d}".format(key))

    plt.title("{:s} : {:d} parameters trained per batch".format(dataset, n_block))
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.tight_layout()
    plt.legend(loc="best")

"""
this method load the dataset and the model, it also trains the model.
It returns the time it took to run the experiment, an historic of test accuracy and training losses
args:
 - model_name : the name of the model to train
 - dataset : the name of the dataset we want to train on
 - optimizer_name : the name of the optimizer we want to use
 - epochs : the number of epochs we want to train
 - batch_size
 - lr : the learning rate
 - weight_decay : the regularization coefficient
 - n_block : the block size to update per batch if we use CD
 - tiny : we use the tiny dataset
"""
def run(model_name, dataset, optimizer_name, epochs=25, batch_size=100,
                 lr=0.001, weight_decay=0,n_block = 4 ,tiny=False):

    optimizer = None
    t1 = time.time()

    if dataset == "FMNIST":
        dataset = "FashionMNIST"
    d_dataset = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}

    train_images, train_targets, test_images, test_targets = load_dataset(d_dataset[dataset], tiny)

    # Random weight initialization and shuffle data
    model = model_name(train_images.size(1))

    # Chooses optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "CD" :
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = CD(model.parameters() ,lr = lr, n_block = n_block, nb_params=total_params)
        

    # Train model
    model, accs, train_loss = train_model(model, optimizer, train_images, train_targets,
                    test_images, test_targets, batch_size, epochs)

    t2 = time.time()
    delay = t2 - t1
    return delay, accs, train_loss