# Implementation of continual learning methods
This repository implements some continual / incremental / lifelong learning methods by PyTorch.

Especially the methods based on **memory replay**.

- [x] iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]
- [x] End2End: End-to-End Incremental Learning. [[paper](https://arxiv.org/abs/1807.09536)]
- [x] DR: Lifelong Learning via Progressive Distillation and Retrospection. [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.html)]
- [x] UCIR: Learning a Unified Classifier Incrementally via Rebalancing. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)]
- [x] BiC: Large Scale Incremental Learning. [[paper](https://arxiv.org/abs/1905.13260)]
- [x] LwM: Learning without Memorizing. [[paper](https://arxiv.org/abs/1811.08051)]
- [x] PODNet: PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning. [[paper](https://arxiv.org/abs/2004.13513)]

## Dependencies
1. torch 1.4.0
2. torchvision 0.5.0
3. tqdm
4. numpy
5. scipy

## Usage

### Run experiment
1. Edit the `config.json` file for global settings.
2. Edit the hyperparameters in the corresponding `.py` file (e.g., `models/icarl.py`).
3. Run:
```bash
python main.py
```

### Add datasets
1. Add corresponding classes to `utils/data.py`.
2. Modify the `_get_idata` function in `utils/data_manager.py`.

## Results

### iCaRL

**CIFAR100**

<img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/iCaRL_cifar100_10.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/iCaRL_cifar100_20.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/iCaRL_cifar100_50.png" width = "325"/>

Average accuracies of CIFAR-100 (iCaRL):
| Increments | Paper reported | Reproduce |
| :--------: | :------------: | :-------: |
| 10 classes | 64.1           | 61.93     |
| 20 classes | 67.2           | 66.24     |
| 50 classes | 68.6           | 67.65     |

### UCIR

**CIFAR100**

<img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_CNN_cifar100_5.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_NCM_cifar100_5.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_CNN_cifar100_10.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_NCM_cifar100_10.png" width = "325"/>

**ImageNet-Subset**

<img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_CNN_imagenet_subset_5.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_NME_imagenet_subset_5.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_CNN_imagenet_subset_10.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/UCIR_NME_imagenet_subset_10.png" width = "325"/>

### BiC

**ImageNet-1000**

|                          | 100  | 200  | 300  | 400  | 500  | 600  | 700  | 800  | 900  | 1000 |
| ------------------------ | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **Paper reported (BiC)** | 94.1 | 92.5 | 89.6 | 89.1 | 85.7 | 83.2 | 80.2 | 77.5 | 75.0 | 73.2 |
| **Reproduce**            | 94.3 | 91.6 | 89.6 | 87.5 | 85.6 | 84.3 | 82.2 | 79.4 | 76.7 | 74.1 |

### PODNet

**CIFAR100**

NME results are shown and the reproduced results are not in line with the reported results. Maybe I missed something...

|     Classifier     |       Steps        |    Reported (%)    |   Reproduced (%)   |
|:-------------------|:------------------:|:------------------:|:------------------:|
|    Cosine (k=1)    |         50         |       56.69        |       55.49        |
|    LSC-CE (k=10)   |         50         |       59.86        |       55.69        |
|   LSC-NCA (k=10)   |         50         |       61.40        |       56.50        |
|    LSC-CE (k=10)   |         25         |       -----        |       59.16        |
|   LSC-NCA (k=10)   |         25         |       62.71        |       59.79        |
|    LSC-CE (k=10)   |         10         |       -----        |       62.59        |
|   LSC-NCA (k=10)   |         10         |       64.03        |       62.81        |
|    LSC-CE (k=10)   |         5          |       -----        |       64.16        |
|   LSC-NCA (k=10)   |         5          |       64.48        |       64.37        |

## Change log
- [x] (2020.6.8) Store the data with list instead of np.array to avoid bugs when the image size is different.
- [x] (2020.7.15) Avoid duplicative selection in constructing exemplars.
- [x] (2020.10.3) Fix the bug of excessive memory usage.
- [x] (2020.10.8) Store the data with np.array instead of Python list to obtain faster I/O.

## Some problems
Q: Why can't I reproduce the results of the paper by this repository?

A: The result of the methods **may be** affected by the incremental order (In my opinion). You can either generate more orders and average their results or increase the number of training iterations (Adjust the hyperparameters).

## References
https://github.com/arthurdouillard/incremental_learning.pytorch
