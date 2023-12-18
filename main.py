import pickle
import time
import numpy as np
import pandas as pd
import torch.optim
from secml.data.loader import CDataLoaderMNIST, CDataLoaderCIFAR10
from secml.ml import CClassifierPyTorch
from secml.adv.attacks import CFoolboxPGDL2
from tinynet import ExpandableCNN_mnist, ExpandableCNN_cifar10, ExpandableCNN2_cifar10, ExpandableFcReLu_mnist, ExpandableFcReLu_cifar10
from secml.array import CArray
from secml.data import CDataset
from performance import plot_performance
from ioaf import compute_indicators
import argparse

from cifar_sampler import cifar_sampler

from folder import CNN_MNIST, FC_MNIST, CNN_CIFAR10, FC_CIFAR10, PLOT_FOLDER_BASE
import config

pd.set_option('display.max_columns', 10)
DEVICE = "cpu"

# expansions = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
# in figure names , we call the old expansions: exp1 and new expansions: exp2
expansions = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
epsilons = np.linspace(0.3, 3, 5)
# indicators_clf = {'epsilons': epsilons, 'overparam' : {}}


def get_model(ds, network):
    if ds == "mnist":
        if network == "cnn":
            model_folder = CNN_MNIST
            clf_names = [f"clf_mnist-cnn_{i}.gz" for i in expansions]
            nn_module = ExpandableCNN_mnist
        elif network == "fcrelu":
            model_folder = FC_MNIST
            clf_names = [f"clf_mnist-fcrelu_{i}.gz" for i in expansions]
            nn_module = ExpandableFcReLu_mnist
    elif ds == "cifar10":
        if network == "cnn":
            model_folder = CNN_CIFAR10
            clf_names = [f"clf_cifar10-cnn_{i}.gz" for i in expansions]
            nn_module = ExpandableCNN_cifar10
        elif network == "fcrelu":
            model_folder = FC_CIFAR10
            clf_names = [f"clf_cifar10-fcrelu_{i}.gz" for i in expansions]
            nn_module = ExpandableFcReLu_cifar10
    return model_folder, clf_names, nn_module 


def load_dataset(ds, train_size, test_size):
    if ds == "mnist":
        tr_dataset = CDataLoaderMNIST().load("training", num_samples=train_size)
        tr_dataset.X /= 255
        ts_dataset = CDataLoaderMNIST().load("testing", num_samples=test_size)
        ts_dataset.X /= 255
    elif ds == "cifar10":
        tr_dataset, ts_dataset = cifar_sampler(train_size, test_size)
        tr_dataset.X /= 255
        ts_dataset.X /= 255
    return (tr_dataset, ts_dataset)



# original script: batch_size: 16, epochs: 20

def pretrain(model_folder, clf_names, tr_dataset, nn_model, input_shape, output_classes, epoch, batch_size, lr):
    clfs = []
    for name, n in zip(clf_names, expansions):
        if not (model_folder / name).exists():
            net = nn_model(expansion=n, out_classes=output_classes)
            print(f"Training network with {sum([i.numel() for i in list(net.parameters())])}")
            net = net.to(DEVICE)
            net.train()
            clf = CClassifierPyTorch(net, input_shape=input_shape,
                                    optimizer=torch.optim.Adam(net.parameters(), lr=lr),
                                    loss=torch.nn.CrossEntropyLoss(), epochs=epoch, batch_size=batch_size)
            clf.fit(tr_dataset.X, tr_dataset.Y)
            print(clf)
            clf.model.eval()
            clf.save(str(model_folder / name))
            clfs.append(clf)
        else:
            clf = CClassifierPyTorch.load(str(model_folder / name))
            clf.model.eval()
            print(f"Loading network with {sum([i.numel() for i in list(clf.model.parameters())])} parameters")
            clfs.append(clf)
    return clfs

def plot_base_performance(clfs, tr_dataset, ts_dataset, ds, network):
    
    parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
    accuracies = []
    if ds == "mnist":
        X = torch.Tensor(ts_dataset.X.tondarray((ts_dataset.X.shape[0], 1, 28, 28)))
    if ds == "cifar10":
        X = torch.Tensor(ts_dataset.X.tondarray((ts_dataset.X.shape[0], 3, 32, 32)))
    plot_performance(clfs, tr_dataset, ts_dataset, CArray(parameters), "Network parameters",
                                title=f"Error rate of {network} on {ds} by varying numbers of parameters",
                                savefig= str(PLOT_FOLDER_BASE / f"{ds}_{network}_base_performance_trainsize{tr_dataset.X.shape[0]}"))
    return parameters

def get_ioaf(parameters, clfs, ts_dataset, folder_pretrained_model, network, steps, stepsize):
    indicators_clf = {'epsilons': epsilons, 'overparam' : {}}
    if not (folder_pretrained_model / "indicators.pkl").exists():
        for p, clf in zip(parameters, clfs):
            indicators_eps = {'indicators': {}, 'robacc': {}}
            for eps in epsilons:
                print(f"Attack {network} with {p} params with PGDL2 eps={eps}")
                indicators_samples = []
                attack_success = []
                for i in range(ts_dataset.X.shape[0]):
                    # print("ts_dataset: ", i)
                    x, y = ts_dataset.X[i, :], ts_dataset.Y[i]
                    indicators = compute_indicators(
                        CFoolboxPGDL2(clf, None, epsilons=eps, abs_stepsize=stepsize, steps=steps, random_start=False), x, y, clf)
                    indicators_samples.append(indicators)
                    attack_success.append(indicators['attack_success'])
                total_success_rate = sum(attack_success) / ts_dataset.X.shape[0]
                print(f"Attack success rate: {total_success_rate}")
                indicators_samples = pd.concat(indicators_samples, axis=0)
                print("Total Mean of indicators samples\n", indicators_samples.mean(axis=0))
                print("\nMean of indicators sample where attack failed\n", indicators_samples[indicators_samples['attack_success'] == False].mean(axis=0)) ## False meaning where the attack sample failed
                indicators_eps['indicators'][eps] = indicators_samples
                indicators_eps['robacc'][eps] = 1 - total_success_rate
            indicators_clf['overparam'][p] = indicators_eps

        with open(str(folder_pretrained_model / "indicators.pkl"), 'wb') as h:
            pickle.dump(indicators_clf, h)



if __name__ == "__main__":
    start_time = "\nScript started...... %s.\n\n " % time.ctime()
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', dest='dataset', choices=['mnist', 'cifar10'], help='Select dataset')
    parser.add_argument('-model', dest='expandable_model', choices=['cnn', 'fcrelu'], help='Select the expandable model')
    parser.add_argument('-train', dest="train_size", choices=['300', '500', '1000', '2000', '3000', '5000', '10000'], help="choose trainind data size")
    parser.add_argument('-ioaf', dest="ioaf_ts_size", help="give a size that is a sample of the test dataset to compute attacks and indicators", required=False)
    
    args = parser.parse_args()
    ds, network, tr_size = args.dataset, args.expandable_model, int(args.train_size)

    if ds == "mnist":
        input_shape = config.MNIST_INPUT_SHAPE
        test_size = config.MNIST_TESTSIZE
    elif ds == "cifar10":
        input_shape = config.CIFAR10_INPUT_SHAPE
        test_size = config.CIFAR10_TESTSIZE

    print(f"\nDATSET: {ds}        MODEL: {network}")
    model_folder, clf_names, nn_module = get_model(ds, network)
    
    print("\n[LOG] Loading dataset...")
    tr_dataset, ts_dataset = load_dataset(ds, tr_size, test_size)

    print(f"\n[LOG] Pretraining with train size: {tr_size} ...")
    clfs = pretrain(model_folder, clf_names, tr_dataset, nn_module, input_shape, config.OUTCLASSES, config.EPOCH1, config.BATCHSIZE1, lr = config.LR1)
    
    print("[LOG] Pretraining complete\n[LOG] Plotting base performance")
    parameters = plot_base_performance(clfs, tr_dataset, ts_dataset, ds, network)

    if args.ioaf_ts_size:
        ioaf_ts_size = int(args.ioaf_ts_size)
        if ds == "mnist":
            ioaf_test = ts_dataset[:ioaf_ts_size, :]
        elif ds == "cifar10":
            ioaf_test_x, ioaf_test_y = ts_dataset.X[:ioaf_ts_size, :], ts_dataset.Y[:ioaf_ts_size]
            ioaf_test = CDataset(ioaf_test_x, ioaf_test_y)
        
        print(f"[LOG] Computing Indicators of Attack Failures on test data of size: {ioaf_test.X.shape[0]}...")
        get_ioaf(parameters, clfs, ioaf_test, model_folder, network, config.STEPS, config.STEPSIZE)

    
    print(f"[LOG] Computing Indicators of Attack Failures on test data of size: {ts_dataset.X.shape[0]}...")
    get_ioaf(parameters, clfs, ts_dataset, model_folder, network, config.STEPS, config.STEPSIZE)


# cifar10-cnn:      batch: 16, epoch: 50, lr: 1e-3, train: variable, test: 10000, hidden: 2: 
# cifar10-fcrelu:   batch: , epoch: , lr: , train: , test: , hidden:
# mnist-cnn:        batch: 16, epoch: 20, lr: 1.e-3, train: 3000, test: 8000, hidden: 2
# mnist-fecrelu:    batch: 16, epoch: 20, lr: 1.e-3, train: 3000, test: 8000, hidden: 5

# _batch16_epoch100_train5000_test10000_2hidden_crossentropy_adam_exp2

#TO DO:

# 1.
# train_size = [300, 500, 1000, 2000, 3000, 5000, 10000]
# test_size = 10000

# 2. try the other cnn model with single layer

# 3. try different loss function
