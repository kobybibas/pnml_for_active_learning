from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10
from nets import MNIST_Net, SVHN_Net, CIFAR10_Net
from query_strategies import (
    RandomSampling,
    LeastConfidence,
    MarginSampling,
    EntropySampling,
    LeastConfidenceDropout,
    MarginSamplingDropout,
    EntropySamplingDropout,
    KMeansSampling,
    KCenterGreedy,
    BALDDropout,
    AdversarialBIM,
    AdversarialDeepFool,
    SingleLayerPnml,
)
from query_strategies.strategy import Strategy


def get_handler(name):
    if name == "MNIST":
        return MNIST_Handler
    elif name == "FashionMNIST":
        return MNIST_Handler
    elif name == "SVHN":
        return SVHN_Handler
    elif name == "CIFAR10":
        return CIFAR10_Handler


def get_dataset(
    name: str, data_dir: str = "../data",
):
    if name == "MNIST":
        return get_MNIST(get_handler(name), data_dir)
    elif name == "FashionMNIST":
        return get_FashionMNIST(get_handler(name), data_dir)
    elif name == "SVHN":
        return get_SVHN(get_handler(name), data_dir)
    elif name == "CIFAR10":
        return get_CIFAR10(get_handler(name), data_dir)
    else:
        raise NotImplementedError


def get_net(name):
    if name == "MNIST":
        return MNIST_Net
    elif name == "FashionMNIST":
        return MNIST_Net
    elif name == "SVHN":
        return SVHN_Net
    elif name == "CIFAR10":
        return CIFAR10_Net
    else:
        raise NotImplementedError


def get_strategy(name: str) -> Strategy:
    if name == "RandomSampling":
        return RandomSampling()
    elif name == "LeastConfidence":
        return LeastConfidence()
    elif name == "MarginSampling":
        return MarginSampling()
    elif name == "EntropySampling":
        return EntropySampling()
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout()
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout()
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout()
    elif name == "KMeansSampling":
        return KMeansSampling()
    elif name == "KCenterGreedy":
        return KCenterGreedy()
    elif name == "BALDDropout":
        return BALDDropout()
    elif name == "AdversarialBIM":
        return AdversarialBIM()
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool()
    elif name == "SingleLayerPnml":
        return SingleLayerPnml()
    else:
        raise NotImplementedError


# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
