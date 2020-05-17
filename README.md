# DCGAN in Tensorflow

本代码基于Carpdm的DCGAN实现（原代码地址：https://github.com/carpedm20/DCGAN-tensorflow），添加如下改善：

1. 原代码中将测试生成数目与生成器噪声维度混用，本代码中将测试图片数目（原generate_test_images参数，改为num_test）与噪声维度参数（添加的input_noise_dim）分离；
2. 源代码使用step计数保存训练权重及sample，改为通过epoch并增加save_epochs参数；
3. 在优化器中添加学习率衰减tf.train.exponential_decay，衰减参数可自行调整，位于train方法开头；



### DCGAN 论文

《深度卷积生成对抗网络》： [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) 。







- 下面介绍一下我使用的软件环境，以及代码文件的基本用法；更多介绍及实验结果可查看原repo。




## 环境配置

下面是我自己使用的版本，可选的两个并未使用。

- Python 3.6
- [Tensorflow 1.14.1](https://github.com/tensorflow/tensorflow/tree/r1.14)
- [SciPy 1.2.1](http://www.scipy.org/install.html)
- [pillow 7.1.2](https://github.com/python-pillow/Pillow)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset





### 使用方法

#### 一、使用MNIST及celebA数据集

##### 1.运行download.py下载数据集：

    $ python download.py mnist celebA

##### 2.开始训练:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --train --crop=False --input_noise_dim=512 --batch_size 64 --epoch 15
    
    $ python main.py --dataset tusimple --output_width=320 --output_height=180 --input_width=320 --input_height=180 --train --crop False --input_noise_dim=256 --batch_size 4 --epoch 10
    
    $ python main.py --dataset pupu --output_width=128 --output_height=128 --input_width=128 --input_height=128 --input_noise_dim 1024 --train --crop=False --batch_size 4 --epoch 40
    
    test pupu      确保宽高、batch_size与权重文件夹相同
    $ python main.py --dataset pupu --output_width=128 --output_height=128 --input_width=128 --input_height=128 --batch_size 4 --input_noise_dim 1024 --train=False --num_test 30
    
    $ test tusimple
    $ python main.py --dataset tusimple --output_width=320 --output_height=180 --input_width=320 --input_height=180 --train=False --input_noise_dim=256 --batch_size 4 --num_test 50
    
    $ python main.py --dataset face --input_height=96 --train --batch_size 16 --sample_steps 20 --epoch 30

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28
    $ python main.py --dataset celebA --input_height=108 --crop

#### 二、使用其它数据集:

    $ mkdir data/DATASET_NAME
    ... 在data/目录中以数据集名称DATASET_NAME为名的文件夹并放入训练数据 ...
    
    $ 如果数据集位于其他目录DATASET_ROOT_DIR：
    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR --train
    
    $ 训练（例）：
    $ python main.py --dataset tusimple --output_width=192 --output_height=108 --input_width=192 --input_height=108 --train --crop=False --input_noise_dim=1024 --batch_size 64 --epoch 15
    
    $ 训练中将以*DATASET_NAME_BATCHSIZE_HEIGHT_WIDTH*的方式保存权重

#### 三、使用预训练模型生成测试图片

训练之后可以使用保存的权重进行生成:

    $ python main.py --dataset tusimple --output_width=192 --output_height=108 --input_width=192 --input_height=108 --train=False --batch_size 64 --num_test 50
    
    $ num_test代表要生成的图片数目，但由于训练中将以*DATASET_NAME_BATCHSIZE_HEIGHT_WIDTH*的方式保存权重，仍需携带batchSize参数。




## 作者

Han Ruizhi / [@Han Zhizhi](https://github.com/HanZhizhi//)
