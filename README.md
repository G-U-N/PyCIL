# PyCIL: A Python Toolbox for Class-Incremental Learning

---

<p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Methods-Reproduced">Methods Reproduced</a> •
  <a href="#Reproduced-Results">Reproduced Results</a> •  
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a> •
  <a href="#Acknowledgments">Acknowledgments</a> •
  <a href="#Contact">Contact</a>
</p>

<div align="center">
<img src="./resources/logo.png" width="200px">
</div>

---



<div align="center">

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)[![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.8-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/) [![method](https://img.shields.io/badge/Reproduced-20-success)]() [![CIL](https://img.shields.io/badge/ClassIncrementalLearning-SOTA-success??style=for-the-badge&logo=appveyor)](https://paperswithcode.com/task/incremental-learning)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=LAMDA.PyCIL&left_color=green&right_color=red)

</div>

Welcome to PyCIL, perhaps the toolbox for class-incremental learning with the **most** implemented methods. This is the code repository for "PyCIL: A Python Toolbox for Class-Incremental Learning" [[paper]](https://arxiv.org/abs/2112.12533) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

    @article{zhou2023pycil,
        author = {Da-Wei Zhou and Fu-Yun Wang and Han-Jia Ye and De-Chuan Zhan},
        title = {PyCIL: a Python toolbox for class-incremental learning},
        journal = {SCIENCE CHINA Information Sciences},
        year = {2023},
        volume = {66},
        number = {9},
        pages = {197101-},
        doi = {https://doi.org/10.1007/s11432-022-3600-y}
      }
    
    @article{zhou2023class,
        author = {Zhou, Da-Wei and Wang, Qi-Wei and Qi, Zhi-Hong and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
        title = {Deep Class-Incremental Learning: A Survey},
        journal = {arXiv preprint arXiv:2302.03648},
        year = {2023}
     }


## What's New
- [2023-09]🌟 We have released [PILOT](https://github.com/sun-hailong/LAMDA-PILOT) toolbox for class-incremental learning with pre-trained models. Have a try!
- [2023-07]🌟 Add [MEMO](https://openreview.net/forum?id=S07feAlQHgM), [BEEF](https://openreview.net/forum?id=iP77_axu0h3), and [SimpleCIL](https://arxiv.org/abs/2303.07338). State-of-the-art methods of 2023!
- [2023-05]🌟 Check out our recent work about [class-incremental learning with vision-language models](https://arxiv.org/abs/2305.19270)!
- [2023-02]🌟 Check out our [rigorous and unified survey](https://arxiv.org/abs/2302.03648) about class-incremental learning, which introduces some memory-agnostic measures with holistic evaluations from multiple aspects!
- [2022-12]🌟 Add FrTrIL, PASS, IL2A, and SSRE.
- [2022-10]🌟 PyCIL has been published in [SCIENCE CHINA Information Sciences](https://link.springer.com/article/10.1007/s11432-022-3600-y) (CCF-A journal). Check out the [official introduction](https://mp.weixin.qq.com/s/h1qu2LpdvjeHAPLOnG478A)!  
- [2022-08]🌟 Add RMM.
- [2022-07]🌟 Add [FOSTER](https://arxiv.org/abs/2204.04662). State-of-the-art method with a single backbone!
- [2021-12]🌟 **Call For Feedback**: We add a <a href="#Awesome-Papers-using-PyCIL">section</a> to introduce awesome works using PyCIL. If you are using PyCIL to publish your work in  top-tier conferences/journals, feel free to [contact us](mailto:zhoudw@lamda.nju.edu.cn) for details!

## Introduction

Traditional machine learning systems are deployed under the closed-world setting, which requires the entire training data before the offline training process. However, real-world applications often face the incoming new classes, and a model should incorporate them continually. The learning paradigm is called Class-Incremental Learning (CIL). We propose a Python toolbox that implements several key algorithms for class-incremental learning to ease the burden of researchers in the machine learning community. The toolbox contains implementations of a number of founding works of CIL, such as EWC and iCaRL, but also provides current state-of-the-art algorithms that can be used for conducting novel fundamental research. This toolbox, named PyCIL for Python Class-Incremental Learning, is open source with an MIT license.

For more information about incremental learning, you can refer to these reading materials:
- A brief introduction (in Chinese) about CIL is available [here](https://zhuanlan.zhihu.com/p/490308909).
- A PyTorch Tutorial to Class-Incremental Learning (with explicit codes and detailed explanations) is available [here](https://github.com/G-U-N/a-PyTorch-Tutorial-to-Class-Incremental-Learning).

## Methods Reproduced

-  `FineTune`: Baseline method which simply updates parameters on new tasks.
-  `EWC`: Overcoming catastrophic forgetting in neural networks. PNAS2017 [[paper](https://arxiv.org/abs/1612.00796)]
-  `LwF`:  Learning without Forgetting. ECCV2016 [[paper](https://arxiv.org/abs/1606.09282)]
-  `Replay`: Baseline method with exemplar replay.
-  `GEM`: Gradient Episodic Memory for Continual Learning. NIPS2017 [[paper](https://arxiv.org/abs/1706.08840)]
-  `iCaRL`: Incremental Classifier and Representation Learning. CVPR2017 [[paper](https://arxiv.org/abs/1611.07725)]
-  `BiC`: Large Scale Incremental Learning. CVPR2019 [[paper](https://arxiv.org/abs/1905.13260)]
-  `WA`: Maintaining Discrimination and Fairness in Class Incremental Learning. CVPR2020 [[paper](https://arxiv.org/abs/1911.07053)]
-  `PODNet`: PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning. ECCV2020 [[paper](https://arxiv.org/abs/2004.13513)]
-  `DER`: DER: Dynamically Expandable Representation for Class Incremental Learning. CVPR2021 [[paper](https://arxiv.org/abs/2103.16788)]
-  `PASS`: Prototype Augmentation and Self-Supervision for Incremental Learning. CVPR2021 [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf)]
-  `RMM`: RMM: Reinforced Memory Management for Class-Incremental Learning. NeurIPS2021 [[paper](https://proceedings.neurips.cc/paper/2021/hash/1cbcaa5abbb6b70f378a3a03d0c26386-Abstract.html)]
-  `IL2A`: Class-Incremental Learning via Dual Augmentation. NeurIPS2021 [[paper](https://proceedings.neurips.cc/paper/2021/file/77ee3bc58ce560b86c2b59363281e914-Paper.pdf)]
-  `SSRE`: Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning. CVPR2022 [[paper](https://arxiv.org/abs/2203.06359)]
-  `FeTrIL`: Feature Translation for Exemplar-Free Class-Incremental Learning. WACV2023 [[paper](https://arxiv.org/abs/2211.13131)]
-  `Coil`: Co-Transport for Class-Incremental Learning. ACM MM2021 [[paper](https://arxiv.org/abs/2107.12654)]
-  `FOSTER`: Feature Boosting and Compression for Class-incremental Learning. ECCV 2022 [[paper](https://arxiv.org/abs/2204.04662)]
-  `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. ICLR 2023 Spotlight [[paper](https://openreview.net/forum?id=S07feAlQHgM)]
-  `BEEF`: BEEF: Bi-Compatible Class-Incremental Learning via Energy-Based Expansion and Fusion. ICLR 2023 [[paper](https://openreview.net/forum?id=iP77_axu0h3)]
-  `SimpleCIL`: Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need. arXiv 2023 [[paper](https://arxiv.org/abs/2303.07338)]

> Intended authors are welcome to contact us to reproduce your methods in our repo. Feel free to merge your algorithm into PyCIL if you are using our codebase!

## Reproduced Results

#### CIFAR-100

<div align="center">
<img src="./resources/cifar100.png" width="900px">
</div>


#### ImageNet-100

<div align="center">
<img src="./resources/ImageNet100.png" width="900px">
</div>

#### ImageNet-100 (Top-5 Accuracy) 

<div align="center">
<img src="./resources/imagenet20st5.png" width="500px">
</div>

> More experimental details and results can be found in our [survey](https://arxiv.org/abs/2302.03648).

## How To Use

### Clone

Clone this GitHub repository:

```
git clone https://github.com/G-U-N/PyCIL.git
cd PyCIL
```

### Dependencies

1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)
6. [quadprog](https://github.com/quadprog/quadprog)
7. [POT](https://github.com/PythonOT/POT)

### Run experiment

1. Edit the `[MODEL NAME].json` file for global settings.
2. Edit the hyperparameters in the corresponding `[MODEL NAME].py` file (e.g., `models/icarl.py`).
3. Run:

```bash
python main.py --config=./exps/[MODEL NAME].json
```

where [MODEL NAME] should be chosen from `finetune`, `ewc`, `lwf`, `replay`, `gem`,  `icarl`, `bic`, `wa`, `podnet`, `der`, etc.

4. `hyper-parameters`

When using PyCIL, you can edit the global parameters and algorithm-specific hyper-parameter in the corresponding json file.

These parameters include:

- **memory-size**: The total exemplar number in the incremental learning process. Assuming there are $K$ classes at the current stage, the model will preserve $\left[\frac{memory-size}{K}\right]$ exemplar per class.
- **init-cls**: The number of classes in the first incremental stage. Since there are different settings in CIL with a different number of classes in the first stage, our framework enables different choices to define the initial stage.
- **increment**: The number of classes in each incremental stage $i$, $i$ > 1. By default, the number of classes per incremental stage is equivalent per stage.
- **convnet-type**: The backbone network for the incremental model. According to the benchmark setting, `ResNet32` is utilized for `CIFAR100`, and `ResNet18` is used for `ImageNet`.
- **seed**: The random seed adopted for shuffling the class order. According to the benchmark setting, it is set to 1993 by default.

Other parameters in terms of model optimization, e.g., batch size, optimization epoch, learning rate, learning rate decay, weight decay, milestone, and temperature, can be modified in the corresponding Python file.

### Datasets

We have implemented the pre-processing of `CIFAR100`, `imagenet100,` and `imagenet1000`. When training on `CIFAR100`, this framework will automatically download it.  When training on `imagenet100/1000`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```
[Here](https://drive.google.com/drive/folders/1RBrPGrZzd1bHU5YG8PjdfwpHANZR_lhJ?usp=sharing) is the file list of ImageNet100 (or say ImageNet-Sub).

## Awesome Papers using PyCIL

### Our Papers

- Learning without Forgetting for Vision-Language Models (**arXiv 2023**) [[paper](https://arxiv.org/abs/2305.19270)]

- Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need (**arXiv 2023**) [[paper](https://arxiv.org/abs/2303.07338)] [[code](https://github.com/zhoudw-zdw/RevisitingCIL)]

- Deep Class-Incremental Learning: A Survey (**arXiv 2023**) [[paper](https://arxiv.org/abs/2302.03648)] [[code](https://github.com/zhoudw-zdw/CIL_Survey/)]

- PILOT: A Pre-Trained Model-Based Continual Learning Toolbox (**arXiv 2023**) [[paper](https://arxiv.org/abs/2309.07117)] [[code](https://github.com/sun-hailong/LAMDA-PILOT)]

- Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration (**NeurIPS 2023**)[[paper](https://arxiv.org/abs/2312.05229)] [[Code](https://github.com/wangkiw/TEEN)]

- BEEF: Bi-Compatible Class-Incremental Learning via Energy-Based Expansion and Fusion (**ICLR 2023**) [[paper](https://openreview.net/forum?id=iP77_axu0h3)] [[code](https://github.com/G-U-N/ICLR23-BEEF/)]

- A model or 603 exemplars: Towards memory-efficient class-incremental learning (**ICLR 2023**) [[paper](https://arxiv.org/abs/2205.13218)] [[code](https://github.com/wangkiw/ICLR23-MEMO/)]

- Few-shot class-incremental learning by sampling multi-phase tasks (**TPAMI 2022**) [[paper](https://arxiv.org/pdf/2203.17030.pdf)] [[code](https://github.com/zhoudw-zdw/TPAMI-Limit)]

- Foster: Feature Boosting and Compression for Class-incremental Learning (**ECCV 2022**) [[paper](https://arxiv.org/abs/2204.04662)] [[code](https://github.com/G-U-N/ECCV22-FOSTER/)]

- Forward compatible few-shot class-incremental learning (**CVPR 2022**) [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)] [[code](https://github.com/zhoudw-zdw/CVPR22-Fact)]

- Co-Transport for Class-Incremental Learning (**ACM MM 2021**) [[paper](https://arxiv.org/abs/2107.12654)] [[code](https://github.com/zhoudw-zdw/MM21-Coil)]

### Other Awesome Works

- Dynamic Residual Classifier for Class Incremental Learning (**ICCV 2023**) [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chavan_Towards_Realistic_Evaluation_of_Industrial_Continual_Learning_Scenarios_with_an_ICCV_2023_paper.pdf)][[code](https://github.com/Vivek9Chavan/RECIL)] 

- Dynamic Residual Classifier for Class Incremental Learning (**ICCV 2023**) [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Dynamic_Residual_Classifier_for_Class_Incremental_Learning_ICCV_2023_paper.pdf)][[code](https://github.com/chen-xw/DRC-CIL)] 

- S-Prompts Learning with Pre-trained Transformers: An Occam's Razor for Domain Incremental Learning (**NeurIPS 2022**) [[paper](https://openreview.net/forum?id=ZVe_WeMold)] [[code](https://github.com/iamwangyabin/S-Prompts)]


## License

Please check the MIT  [license](./LICENSE) that is listed in this repository.

## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [Continual-Learning-Reproduce](https://github.com/zhchuu/continual-learning-reproduce)
- [GEM](https://github.com/hursung1/GradientEpisodicMemory)
- [FACIL](https://github.com/mmasana/FACIL)

The training flow and data configurations are based on Continual-Learning-Reproduce. The original information of the repo is available in the base branch.


## Contact

If there are any questions, please feel free to  propose new features by opening an issue or contact with the author: **Da-Wei Zhou**([zhoudw@lamda.nju.edu.cn](mailto:zhoudw@lamda.nju.edu.cn)) and **Fu-Yun Wang**(wangfuyun@smail.nju.edu.cn). Enjoy the code.


## Star History 🚀

[![Star History Chart](https://api.star-history.com/svg?repos=G-U-N/PyCIL&type=Date)](https://star-history.com/#G-U-N/PyCIL&Date)

