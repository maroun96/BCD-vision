from convnet import Net
from helpers import run, plot
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", help="Name of the dataset to load")
    parser.add_argument("--optim", default="SGD", help="Choose the optimizer to use")
    parser.add_argument("--epochs", type=int, default=25, help="Choose the number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for regularization")
    parser.add_argument("--tiny", action="store_true", help="Reduces dataset")

    parser.add_argument("--n_block", type= int, default= 4, help="Specify this paramter if working with CD")
    parser.add_argument("--batch_size", type=int, default=100, help="Choose the batch size")
    parser.add_argument("--run_all", action="store_true", help="You'll run the four optimization experiments for the choosen dataset")
    parser.add_argument("--plot", action="store_true", help="Plots results in pkl files")

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    optimizers = ["SGD", "CD"]
    assert args.optim in optimizers
    model = Net
    default_nblock = sum(p.numel() for p in Net(1).parameters())

    #We reproduce our plots
    if args.plot:
        with open("{:s}_historic.pkl".format(args.dataset), "rb") as f:
            all_losses = pickle.load(f)
        with open("{:s}_accs.pkl".format(args.dataset), "rb") as f:
            all_accs = pickle.load(f)

        plt.figure(figsize=(12, 12))
        table = [default_nblock, 2500, 500, 100]

        matplotlib.rcParams.update({'font.size': 13})
        for i, train_losses in enumerate(all_losses):
            plot(args.dataset, table[i], train_losses, 2, 2, i + 1)
        plt.savefig("plots/{:s}_allrun.png".format(args.dataset))

    else:
        if args.optim == "SGD":
            args.n_block = default_nblock

        # We run all our experiments
        if args.run_all:
            plt.figure(figsize=(12,12))
            final_print = ""
            all_losses = []
            all_accs = []
            for i, n_block in enumerate([default_nblock, 2500, 500, 100]):
                train_losses = {}
                accs_dict = {}
                for batch_size in [10000, 1000, 200, 50]:
                    exp_name = "Experience on {:s} with {:s} | n_block={:d} | batch_size={:d}".format(args.dataset, args.optim, n_block,batch_size)
                    final_print = final_print + exp_name
                    print("\n" + exp_name)
                    delay, accs, train_loss = run(model, args.dataset, args.optim, args.epochs, batch_size,
                                                    args.lr, args.weight_decay,n_block, args.tiny)
                    train_losses[batch_size] = train_loss
                    accs_dict[batch_size] = accs
                    results = "Experience took {:d} s | Final Accuracy: {:.2f}%".format(int(delay), accs[-1])
                    print(results)
                    final_print = final_print + " : " + results + "\n"

                plot(args.dataset, n_block, train_losses, 2, 2, i+1)
                all_losses.append(train_losses)
                all_accs.append(accs_dict)

            with open("{:s}_historic.pkl".format(args.dataset), "wb") as f:
                pickle.dump(all_losses, f)
            with open("{:s}_accs.pkl".format(args.dataset), "wb") as f:
                pickle.dump(all_accs, f)
            plt.savefig("plots/{:s}_allrun.png".format(args.dataset))
            print(final_print)

        #We run one experiment with parameters choosen in commands
        else:
            plt.figure(figsize=(6, 6))
            print("Experience on {:s} with {:s}".format(args.dataset, args.optim))
            delay, accs, train_loss = run(model, args.dataset, args.optim, args.epochs, args.batch_size,
                                            args.lr, args.weight_decay,args.n_block, args.tiny)
            print("Experience took {:d} s | Final Accuracy: {:.2f}%".format(int(delay), accs[-1]))
            plot(args.dataset, args.n_block, {args.batch_size : train_loss})
            plt.savefig("plots/{:s}_{:d}.png".format(args.dataset, args.n_block))