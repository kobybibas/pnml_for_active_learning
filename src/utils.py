from data import (
    get_CIFAR10,
    get_EMNIST,
    get_FashionMNIST,
    get_MNIST,
    get_MNIST_C,
    get_MNIST_OOD,
    get_SVHN,
    get_CIFAR10_OOD,
)
from nets import CIFAR10_Net, EMNIST_Net, MNIST_Net, SVHN_Net
from query_strategies import (
    AdversarialBIM,
    AdversarialDeepFool,
    BALDDropout,
    DropoutGenie,
    DropoutPnml,
    DropoutPnmlPrior,
    EntropySampling,
    EntropySamplingDropout,
    KCenterGreedy,
    KMeansSampling,
    LeastConfidence,
    LeastConfidenceDropout,
    MarginSampling,
    MarginSamplingDropout,
    RandomSampling,
)
from query_strategies.strategy import Strategy


def get_dataset(
    name: str,
    data_dir: str = "../data",
    val_set_size: int = 1024,
    dataset_limit: int = -1,
):
    if name == "MNIST":
        return get_MNIST(data_dir, val_set_size, dataset_limit)
    elif name == "MNIST_C":
        return get_MNIST_C(data_dir, val_set_size, dataset_limit)
    elif name == "MNIST_OOD":
        return get_MNIST_OOD(data_dir, val_set_size, dataset_limit)
    elif name == "EMNIST":
        return get_EMNIST(data_dir, val_set_size, dataset_limit)
    elif name == "FashionMNIST":
        return get_FashionMNIST(data_dir, val_set_size, dataset_limit)
    elif name == "SVHN":
        return get_SVHN(data_dir, val_set_size, dataset_limit)
    elif name == "CIFAR10":
        return get_CIFAR10(data_dir, val_set_size, dataset_limit)
    elif name == "CIFAR10_OOD":
        return get_CIFAR10_OOD(data_dir, val_set_size, dataset_limit)
    else:
        raise NotImplementedError


def get_net(name):
    if name in ("MNIST", "MNIST_C", "MNIST_OOD", "FashionMNIST"):
        return MNIST_Net
    elif name == "EMNIST":
        return EMNIST_Net
    elif name == "SVHN":
        return SVHN_Net
    elif name in ("CIFAR10", "CIFAR10_OOD"):
        return CIFAR10_Net
    else:
        raise NotImplementedError


def get_strategy(
    name: str,
    n_drop: int = 10,
    query_batch_size: int = -1,
    unlabeled_pool_size: int = -1,
    test_set_size: int = -1,
    temperature: float = 1.0,
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
            temperature=temperature,
        )
    elif name == "DropoutPnmlPrior":
        return DropoutPnmlPrior(
            n_drop=n_drop,
            query_batch_size=query_batch_size,
            unlabeled_pool_size=unlabeled_pool_size,
            test_set_size=test_set_size,
            temperature=temperature,
        )
    elif name == "DropoutGenie":
        return DropoutGenie(
            n_drop=n_drop,
            query_batch_size=query_batch_size,
            unlabeled_pool_size=unlabeled_pool_size,
            test_set_size=test_set_size,
            temperature=temperature,
        )

    else:
        raise NotImplementedError


# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
