import pickle

import numpy as np
import pandas as pd
import torch.optim
from secml.adv.attacks import CFoolboxPGDL2
from secml.array import CArray
from secml.data.loader import CDataLoaderMNIST
from secml.ml import CClassifierPyTorch

from config import FCRELU_PRETRAINED_MNIST, PLOT_FOLDER
from ioaf import compute_indicators
from performance import plot_performance
from tinynet import Expandable2LinearReLu

pd.set_option('display.max_columns', 10)

DEVICE = "cpu"
STEPS = 100
DIGITS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
STEPSIZE = 0.05

input_shape = (1, 28, 28)
model_folder = FCRELU_PRETRAINED_MNIST
tr_dataset = CDataLoaderMNIST().load(
    "training", num_samples=3000, digits=DIGITS
)
tr_dataset.X /= 255
expansions = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]

print("Training Data Size: ", tr_dataset.X.shape)

clf_names = [f"clf_2fcRelu_{i}.gz" for i in expansions]
clfs = []

epsilons = np.linspace(0.3, 3, 5)

for name, n in zip(clf_names, expansions):
    if not (model_folder / name).exists():
        net = Expandable2LinearReLu(expansion=n, mnist=True, colored=False, out_classes=len(DIGITS))
        print(f"Training network with {sum([i.numel() for i in list(net.parameters())])}")
        net = net.to(DEVICE)
        net.train()
        clf = CClassifierPyTorch(net, input_shape=input_shape,
                                 optimizer=torch.optim.Adam(net.parameters(), lr=1.e-3),
                                 loss=torch.nn.CrossEntropyLoss(), epochs=20, batch_size=16)
        clf.fit(tr_dataset.X, tr_dataset.Y)
        print(clf)
        clf.model.eval()
        clf.save(str(model_folder / name))
        clfs.append(clf)
    else:
        clf = CClassifierPyTorch.load(str(model_folder / name))
        clf.model.eval()
        print(f"Loading network with {sum([i.numel() for i in list(clf.model.parameters())])}")
        clfs.append(clf)

ts_dataset = CDataLoaderMNIST().load("testing", num_samples=100, digits=DIGITS)
ts_dataset.X /= 255
parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
accuracies = []
X = torch.Tensor(ts_dataset.X.tondarray((ts_dataset.X.shape[0], 1, 28, 28)))
plot_performance(clfs, tr_dataset, ts_dataset, CArray(parameters), "Network parameters",
                             title="Accuracy of 2 layered Relu network by varying numbers of parameters", savefig=str(PLOT_FOLDER / "mnist_fcrelu_base_performance"))
indicators_clf = {'epsilons': epsilons, 'overparam' : {}}
if not (FCRELU_PRETRAINED_MNIST / "indicators.pkl").exists():
    for p, clf in zip(parameters, clfs):
        indicators_eps = {'indicators': {}, 'robacc': {}}
        for eps in epsilons:
            print(f"Attack 2L-Relu network with {p} params with PGDL2 eps={eps}")
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

    with open(str(FCRELU_PRETRAINED_MNIST / "indicators.pkl"), 'wb') as h:
        pickle.dump(indicators_clf, h)
