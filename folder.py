import pathlib as pl

BASE = pl.Path(__file__).parent.resolve()

DATA = BASE / "data"
SUBSETCIFAR = DATA / "cifar10"
DATA.mkdir(exist_ok=True)
SUBSETCIFAR.mkdir(exist_ok=True)


PLOT_FOLDER = BASE / "plots"
# PRL = PLOT_FOLDER / "prl"
PLOT_FOLDER.mkdir(exist_ok=True)

MNIST_BAR_INDICATOR = PLOT_FOLDER / "mnist_bar_indicator"
CIFAR10_BAR_INDICATOR = PLOT_FOLDER / "cifar10_bar_indicator"
MNIST_BAR_INDICATOR.mkdir(exist_ok=True)
CIFAR10_BAR_INDICATOR.mkdir(exist_ok=True)

MNIST_CNN_INDICATOR_PLOT = MNIST_BAR_INDICATOR / "cnn"
MNIST_FCRELU_INDICATOR_PLOT = MNIST_BAR_INDICATOR / "fcrelu"
MNIST_FCRELU100_INDICATOR_PLOT = MNIST_BAR_INDICATOR / "fcrelu100"
CIFAR10_CNN_INDICATOR_PLOT = CIFAR10_BAR_INDICATOR / "cnn"
CIFAR10_FCRELU_INDICATOR_PLOT = CIFAR10_BAR_INDICATOR / "fcrelu"

PRETRAINED = BASE / "pretrained"
DATA_FOLDER = BASE / "data"


PLOT_FOLDER.mkdir(exist_ok=True)
PRETRAINED.mkdir(exist_ok=True)
# PRL.mkdir(exist_ok=True)
MNIST_CNN_INDICATOR_PLOT.mkdir(exist_ok=True)
MNIST_FCRELU_INDICATOR_PLOT.mkdir(exist_ok=True)
MNIST_FCRELU100_INDICATOR_PLOT.mkdir(exist_ok=True)
CIFAR10_CNN_INDICATOR_PLOT.mkdir(exist_ok=True)
CIFAR10_FCRELU_INDICATOR_PLOT.mkdir(exist_ok=True)


# MNIST_MODEL = PRETRAINED / "mnist"
# CIFAR10_MODEL = PRETRAINED / "cifar10"

# MNIST_MODEL.mkdir(exist_ok=True)
# CIFAR10_MODEL.mkdir(exist_ok=True)

# TINYNET_PRETRAINED_MNIST = MNIST_MODEL / "tinynet"
# TINYNET_PRETRAINED_CIFAR10 = CIFAR10_MODEL / "tinynet"

# FCRELU_PRETRAINED_MNIST = MNIST_MODEL / "2fcRelu"
# FCRELU_PRETRAINED_CIFAR10 = CIFAR10_MODEL / "2fcRelu"

# TINYNET_PRETRAINED_MNIST.mkdir(exist_ok=True)
# TINYNET_PRETRAINED_CIFAR10.mkdir(exist_ok=True)
# FCRELU_PRETRAINED_MNIST.mkdir(exist_ok=True)
# FCRELU_PRETRAINED_CIFAR10.mkdir(exist_ok=True)

# REG_TINYNET_PRETRAINED_MNIST = TINYNET_PRETRAINED_MNIST / "standard"
# REG_TINYNET_PRETRAINED_CIFAR10 = TINYNET_PRETRAINED_CIFAR10 / "standard"

# REG_TINYNET_PRETRAINED_MNIST.mkdir(exist_ok=True)
# REG_TINYNET_PRETRAINED_CIFAR10.mkdir(exist_ok = True)

BASEMODEL = BASE / "base_model"
BASEMODEL.mkdir(exist_ok=True)

MNIST_BASE = BASEMODEL / "mnist"
CIFAR10_BASE = BASEMODEL / "cifar10"
# PLOT_FOLDER_BASE = BASEMODEL / "plots"
MNIST_BASE.mkdir(exist_ok=True)
CIFAR10_BASE.mkdir(exist_ok=True)
# PLOT_FOLDER_BASE.mkdir(exist_ok=True)

CNN_MNIST = MNIST_BASE / "cnn"
FC_MNIST = MNIST_BASE / "fcrelu"
CNN_CIFAR10 = CIFAR10_BASE / "cnn"
FC_CIFAR10 = CIFAR10_BASE / "fcrelu"
CNN_MNIST.mkdir(exist_ok=True)
FC_MNIST.mkdir(exist_ok=True)
CNN_CIFAR10.mkdir(exist_ok=True)
FC_CIFAR10.mkdir(exist_ok=True)