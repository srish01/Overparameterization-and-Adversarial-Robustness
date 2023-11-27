# Overparametrization experiments

## Installation
Create environment:
```bash
conda create -n overparam python=3.10
conda activate overparam
pip install -r requirements.txt
```

## Experiments

Run `mnist_indicators_robustness` to train, attack, and compute the indicators on a tiny CNN trained on all MNIST digits.
Have a look at the script to read the attacks parameters.

Run `plot_ts_error_indicator` to plot accuracy, robustness, and indicators.

To extend to CIFAR, copy `mnist_indicators_robustness`, adapt the folders where it will save the results (look at `config.py`).
Then, apply the same functions from `plot_ts_error_indicator`.