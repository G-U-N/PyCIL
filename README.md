# Implementation of continual learning methods
This repository implements some continual / incremental / lifelong learning methods by PyTorch.

Especially the methods based on **memory replay**.

- [x] iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]
- [x] End2End: End-to-End Incremental Learning. [[paper](https://arxiv.org/abs/1807.09536)]
- [x] DR: Lifelong Learning via Progressive Distillation and Retrospection. [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.html)]

## Dependencies
1. torch 1.4.0
2. torchvision 0.5.0
3. tqdm
4. numpy
5. scipy

## Usage
1. Edit the *config.json* file.
2. Run:
```bash
python main.py
```

## Results
<img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/cifar100_10.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/cifar100_20.png" width = "325"/><img src="https://github.com/zhchuu/continual-learning-reproduce/blob/master/resources/cifar100_50.png" width = "325"/>

Average accuracies of CIFAR-100:
| Increments | Paper reported | Reproduce |
| :--------: | :------------: | :-------: |
| 10 classes | 64.1           | 61.93     |
| 20 classes | 67.2           | 66.24     |
| 50 classes | 68.6           | 67.65     |

## Change log
- [x] (2020.6.8) Store data with list instead of np.array to avoid bugs when the image size is different.

## Some problems
Q: Why can't I reproduce the results of the paper by this repository?

A: The result of the methods **may be** affected by the incremental order (In my opinion). You can either generate more orders and average their results or increase the number of training iterations (Adjust the hyperparameters).

## References
https://github.com/arthurdouillard/incremental_learning.pytorch
