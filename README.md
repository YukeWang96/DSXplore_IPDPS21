<!-- # Train CIFAR10 with PyTorch -->

# DSXplore: Optimizing Convolutional NeuralNetworks via Sliding-Channel Convolutions
Accepted at IPDPS-2021 [[arxiv](https://arxiv.org/abs/2101.00745)]

> Author: **Yuke Wang**, Boyuan Feng, and Yufei Ding \
> Email: yuke_wang@cs.ucsb.edu

+ Dependency: `Python 3.7`, `nvcc 11.1`.
+ Install Conda to set up an virtual environment [Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart).
+ Install Pytorch with GPU support [Toturial](https://pytorch.org/get-started/locally/).
+ Go to ``SCC_conv/``, then ``python setup.py install``.


# Run
+ Avaiable Models ``[model name]``: ``VGG11, VGG13, VGG16, VGG19, ResNet18, ResNet34, ResNet50, MobileNet``.
+ Avaiable groups ``[num_group]``: ``1,2,4,8``.
+ Avaiable overlap ``[overlap_ratio]``: ``0.25,0.33,0.50,0.75``.
+ Then execute ``python main.py --model [Model Name] --groups [num_group] --overlap [overlap_ratio]`` .
+ A detailed example of changing backend convolution implementation is illustrated in `models/vgg.py`.


# Reference
[Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar.git)