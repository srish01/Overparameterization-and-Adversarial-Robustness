import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from secml.array import CArray
from secml.data import CDataset
from secml.ml import CClassifier
from secml.ml.peval.metrics import CMetricAccuracy

from folder import PLOT_FOLDER


def performance_score_model(classifier: CClassifier, tr_dataset: CDataset, ts_dataset: CDataset):
    tr_y_pred = classifier.predict(tr_dataset.X)
    ts_y_pred = classifier.predict(ts_dataset.X)
    tr_acc = CMetricAccuracy().performance_score(tr_dataset.Y, tr_y_pred)
    ts_acc = CMetricAccuracy().performance_score(ts_dataset.Y, ts_y_pred)
    tr_error = 1 - tr_acc
    ts_error = 1 - ts_acc
    return tr_acc, ts_acc, tr_error, ts_error


def log_mse_loss(clf: CClassifier, data: CDataset):
    scores = clf.decision_function(data.X)
    predictions = softmax(scores.tondarray(), axis=1)
    labels = data.Y.tondarray()
    y = np.zeros((labels.size, labels.max() + 1))
    y[np.arange(labels.size), labels] = 1
    loss = np.log(np.mean([(f - yi)**2 for f, yi in zip(predictions, y)]))
    print(f"Loss: {loss}")
    return loss


# def test_models(classifiers: list[CClassifier], tr_dataset: CDataset, ts_dataset: CDataset):
def test_models(classifiers, tr_dataset, ts_dataset):
    ts_errors = []
    tr_errors = []
    for c in classifiers:
        tr_acc, ts_acc, tr_err, ts_err = performance_score_model(c, tr_dataset, ts_dataset)
        ts_errors.append(ts_err)
        tr_errors.append(tr_err)
    ts_errors = CArray(ts_errors)
    tr_errors = CArray(tr_errors)

    return tr_errors, ts_errors


# def test_model_log_mse(classifiers: list[CClassifier], tr_dataset: CDataset, ts_dataset: CDataset):
def test_model_log_mse(classifiers, tr_dataset, ts_dataset):
    ts_errors = []
    tr_errors = []
    for c in classifiers:
        tr, ts = log_mse_loss(c, tr_dataset), log_mse_loss(c, ts_dataset)
        ts_errors.append(ts)
        tr_errors.append(tr)
    ts_errors = CArray(ts_errors)
    tr_errors = CArray(tr_errors)

    return tr_errors, ts_errors

# classifiers: list[CClassifier]
def plot_performance_log_mse(classifiers,
                             tr_dataset: CDataset,
                             ts_dataset: CDataset,
                             x_axis: CArray,
                             xlabel: str = "",
                             title: str = "",
                             savefig: str = None):
    tr_error, ts_error = test_model_log_mse(classifiers, tr_dataset, ts_dataset)
    plt.semilogx(x_axis.tondarray(), tr_error.tondarray(), label="tr error", c='r')
    plt.semilogx(x_axis.tondarray(), ts_error.tondarray(), label="ts error", c='b')
    plt.xlabel(xlabel)
    plt.ylabel("log MSE Error")
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig(str(PLOT_FOLDER / f"{savefig}.pdf"))
    plt.show()


def plot_performance(classifiers,
                             tr_dataset: CDataset,
                             ts_dataset: CDataset,
                             x_axis: CArray,
                             xlabel: str = "",
                             title: str = "",
                             savefig: str = None):
    
    print("classifier: ", type(classifiers))
    print("tr_dataset: ", type(tr_dataset))
    print("ts_dataset: ", type(ts_dataset))
    tr_error, ts_error = test_models(classifiers, tr_dataset, ts_dataset)
    plt.plot(x_axis.tondarray(), tr_error.tondarray(), label="tr error", c='r')
    plt.plot(x_axis.tondarray(), ts_error.tondarray(), label="ts error", c='b')
    plt.xlabel(xlabel)
    plt.ylabel("Errors")    # previously "Accuracies" but we are actually plotting errors
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig(str(PLOT_FOLDER / f"{savefig}.jpg"))
    plt.show()