# Deep Individual Active Learning: Safeguarding against Out-of-Distribution Challenges in Neural Networks

Abstract
> Active learning (AL) is a paradigm focused on purposefully selecting training data to enhance a model’s performance by minimizing the need for annotated samples. Typically, strategies assume that the training pool shares the same distribution as the test set, which is not always valid in privacy-sensitive applications where annotating user data is challenging. In this study, we operate within an individual setting and leverage an active learning criterion which selects data points for labeling based on minimizing the min-max regret on a small unlabeled test set sample. Our key contribution lies in the development of an efficient algorithm, addressing the challenging computational complexity associated with approximating this criterion for neural networks. Notably, our results show that, especially in the presence of out-of-distribution data, the proposed algorithm substantially reduces the required training set size by up to 15.4%, 11%, and 35.1% for CIFAR10, EMNIST, and MNIST datasets, respectively.


[paper link](https://www.mdpi.com/1099-4300/26/2/129)


# Setup

```
conda create -n pnml_for_active_learning
conda activate pnml_for_active_learning
pip install -r requirements.txt
pip install tmuxp
```

# Execute
All executable experiment scrips are located in bash_scripts.
An example of executing one of them:
```
tmuxp load ./mysession.yaml
bash_scripts/execute_CIFAR10.yaml
```


# Cite

```
@Article{e26020129,
AUTHOR = {Shayovitz, Shachar and Bibas, Koby and Feder, Meir},
TITLE = {Deep Individual Active Learning: Safeguarding against Out-of-Distribution Challenges in Neural Networks},
JOURNAL = {Entropy},
VOLUME = {26},
YEAR = {2024},
NUMBER = {2},
ARTICLE-NUMBER = {129},
URL = {https://www.mdpi.com/1099-4300/26/2/129},
ISSN = {1099-4300},
DOI = {10.3390/e26020129}
}
```




