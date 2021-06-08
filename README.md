# BCD-vision
This repository proposes pytorch code to train a convolutional model on MNIST or FMNIST with Stochastic Gradient Descent (SGD) or Block Coordinate Descent (BCD). It aims to show the different behaviours of the training related to the schochaticity and the number of parameters we update per batch.

## How to run ?


There are several parameters available:

```
--dataset       : Either MNIST or FashionMNIST, choose the dataset you want the model to train on.
--optim         : Either SGD or CD, choose the optimization method.
--epochs        : Specify the number of epochs you want to train.
--lr            : Specify the initial learning rate for you optimization steps.
--weight_decay  : Specify the regularization term.
--tiny          : If you add this argument, the training will be done on a dataset of 5000 training and testing samples.
--n_block       : Specify the number of epochs you want to train.
--lr            : Specify the initial learning rate for you optimization steps.
--weight_decay  : Specify the regularization term.
--tiny          : If you add this argument, the training will be done on a dataset of 5000 training and testing samples.
--n_block       : Specify the number of parameters you want to update at every batch.
--batch_size    : Specify the batch size.
--run_all       : If you add this argument, you will run all our experiments (takes a bit less than two hours for one dataset).
```

Below, you can find an exemple to run typical experiment. After every epoch, you will get the training loss and the accuracy on testing dataset. At the end, you'll get the final accuracy and you can find a plot of the training loss against the epoch number in `plots` folder.

```
python main.py --dataset MNIST --optim CD --lr 0.25 --n_block 1000 --batch_size 200
```

## How ro reproduce our results ?

To reproduce all our results, you need to run the code below. In addition to the outpus listed above, it will also output a summary of all experiments at the end. You can find all plots in the `plots` folder.

```
python main.py --dataset MNIST --optim CD --lr 0.25 --epochs 100 --run_all
python main.py --dataset FashionMNIST --optim CD --lr 0.25 --epochs 100 --run_all
```
Be careful, these runs will take three hours each.

If you only want to reproduce our plots, run the following commands:

```
python main.py --dataset MNIST --plot
python main.py --dataset FMNIST --plot
```

It will read our experiments results in the .pkl files and reproduce our plots.