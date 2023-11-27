import pickle

import numpy as np
import pandas as pd
import torch.optim
from secml.array import CArray
from secml.adv.attacks import CFoolboxPGDL2
from secml.data.loader import CDataLoaderCIFAR10, CDataLoaderMNIST
from secml.ml import CClassifierPyTorch

from config import REG_TINYNET_PRETRAINED_CIFAR10, PLOT_FOLDER
from performance import plot_performance
from ioaf import compute_indicators
from tinynet import ExpandableTinyNet, Expandable2LinearReLu

pd.set_option('display.max_columns', 10)

DEVICE = "cpu"
STEPS = 100
STEPSIZE = 0.05
TRAIN_SIZE = 3000
TEST_SIZE = 10

input_shape = (3, 32, 32)
model_folder = REG_TINYNET_PRETRAINED_CIFAR10
cifar_ds = CDataLoaderCIFAR10().load()

tr_dataset = cifar_ds[0][:TRAIN_SIZE, :]

tr_dataset.X /= 255

print("Training Data size: ", tr_dataset.X.shape)
print("Training Data Sabel Size Y: ", tr_dataset.Y.shape)
print("-----------\n Unique label in the dataset", set(tr_dataset.Y))

expansions = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]

clf_names = [f"clf_cnn_{i}.gz" for i in expansions]
clfs = []

epsilons = np.linspace(0.3, 3, 5)

for name, n in zip(clf_names, expansions):
    if not (model_folder / name).exists():
        net = ExpandableTinyNet(expansion=n, mnist=False, colored=True, out_classes=10)
        print(f"Training network with {sum([i.numel() for i in list(net.parameters())])}")
        net = net.to(DEVICE)
        net.train()
        clf = CClassifierPyTorch(net, input_shape=input_shape,
                                 optimizer=torch.optim.Adam(net.parameters(), lr=1.e-3),
                                 loss=torch.nn.CrossEntropyLoss(), epochs=20, batch_size=16)
        print(clf)
        clf.fit(tr_dataset.X, tr_dataset.Y)
        clf.model.eval()
        clf.save(str(model_folder / name))
        clfs.append(clf)
    else:
        clf = CClassifierPyTorch.load(str(model_folder / name))
        clf.model.eval()
        print(f"Loading network with {sum([i.numel() for i in list(clf.model.parameters())])}")
        clfs.append(clf)

print(clfs)

ts_dataset = cifar_ds[1][:TEST_SIZE, :]
ts_dataset.X/= 255

parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]

accuracies = []

X = torch.Tensor(ts_dataset.X.tondarray((ts_dataset.X.shape[0], 3, 32, 32)))
plot_performance(clfs, tr_dataset, ts_dataset, CArray(parameters), "Network parameters",
                             title="Accuracy of CNN by varying numbers of parameters", savefig=str(PLOT_FOLDER / "cifar10_cnn_base_performance"))
indicators_clf = {'epsilons': epsilons, 'overparam' : {}}
if not (REG_TINYNET_PRETRAINED_CIFAR10 / "indicators.pkl").exists():
    for p, clf in zip(parameters, clfs):
        indicators_eps = {'indicators': {}, 'robacc': {}}
        for eps in epsilons:
            print(f"Attack CNN with {p} params with PGDL2 eps={eps}")
            indicators_samples = []
            attack_success = []
            for i in range(ts_dataset.X.shape[0]):
                x, y = ts_dataset.X[i, :], ts_dataset.Y[i]
                indicators = compute_indicators(
                    CFoolboxPGDL2(clf, None, epsilons=eps, abs_stepsize=STEPSIZE, steps=STEPS, random_start=False), x,
                    y, clf)
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

    with open(str(REG_TINYNET_PRETRAINED_CIFAR10 / "indicators.pkl"), 'wb') as h:
        pickle.dump(indicators_clf, h)