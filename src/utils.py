from data import (
    get_CIFAR10,
    get_CINIC10,
    get_EMNIST,
    get_FashionMNIST,
    get_MNIST,
    get_SVHN,
    get_MNIST_C,
    get_MNIST_OOD,
)
from handlers import CIFAR10_Handler, CINIC10_Handler, MNIST_Handler, SVHN_Handler
from nets import (
    CIFAR10_Net,
    CINIC10_Net,
    EMNIST_Net,
    MNIST_Net,
    SVHN_Net,
    MNIST_OOD_Net,
)
from query_strategies import (
    AdversarialBIM,
    AdversarialDeepFool,
    BALDDropout,
    DropoutPnml,
    EntropySampling,
    EntropySamplingDropout,
    KCenterGreedy,
    KMeansSampling,
    LeastConfidence,
    LeastConfidenceDropout,
    MarginSampling,
    MarginSamplingDropout,
    RandomSampling,
    DropoutGenie,
)
from query_strategies.strategy import Strategy


def get_handler(name):
    if name == "MNIST":
        return MNIST_Handler
    elif name == "MNIST_C":
        return MNIST_Handler
    elif name == "MNIST_OOD":
        return MNIST_Handler
    elif name == "EMNIST":
        return MNIST_Handler
    elif name == "FashionMNIST":
        return MNIST_Handler
    elif name == "SVHN":
        return SVHN_Handler
    elif name == "CIFAR10":
        return CIFAR10_Handler
    elif name == "CINIC10":
        return CINIC10_Handler


def get_dataset(name: str, data_dir: str = "../data", validation_set_size: int = 1024):
    if name == "MNIST":
        return get_MNIST(get_handler(name), data_dir, validation_set_size)
    elif name == "MNIST_C":
        return get_MNIST_C(get_handler(name), data_dir, validation_set_size)
    elif name == "MNIST_OOD":
        return get_MNIST_OOD(get_handler(name), data_dir, validation_set_size)
    elif name == "EMNIST":
        return get_EMNIST(get_handler(name), data_dir, validation_set_size)
    elif name == "FashionMNIST":
        return get_FashionMNIST(get_handler(name), data_dir)
    elif name == "SVHN":
        return get_SVHN(get_handler(name), data_dir)
    elif name == "CIFAR10":
        return get_CIFAR10(get_handler(name), data_dir)
    elif name == "CINIC10":
        return get_CINIC10(get_handler(name), data_dir)
    else:
        raise NotImplementedError


def get_net(name):
    if name == "MNIST":
        return MNIST_Net
    elif name == "MNIST_C":
        return MNIST_Net
    elif name == "EMNIST":
        return EMNIST_Net
    elif name == "FashionMNIST":
        return MNIST_Net
    elif name == "SVHN":
        return SVHN_Net
    elif name == "CIFAR10":
        return CIFAR10_Net
    elif name == "CINIC10":
        return CINIC10_Net
    elif name == "MNIST_OOD":
        return MNIST_OOD_Net
    else:
        raise NotImplementedError


def get_strategy(
    name: str,
    n_drop: int = 10,
    query_batch_size: int = -1,
    unlabeled_pool_size: int = -1,
    test_set_size: int = -1,
) -> Strategy:
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
        return BALDDropout(
            n_drop=n_drop,
            unlabeled_pool_size=unlabeled_pool_size,
        )
    elif name == "AdversarialBIM":
        return AdversarialBIM()
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool()
    elif name == "DropoutPnml":
        return DropoutPnml(
            n_drop=n_drop,
            query_batch_size=query_batch_size,
            unlabeled_pool_size=unlabeled_pool_size,
            test_set_size=test_set_size,
        )
    elif name == "DropoutGenie":
        return DropoutGenie(
            n_drop=n_drop,
            query_batch_size=query_batch_size,
            unlabeled_pool_size=unlabeled_pool_size,
            test_set_size=test_set_size,
        )

    else:
        raise NotImplementedError


# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
