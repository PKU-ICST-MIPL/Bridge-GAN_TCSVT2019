## Introduction
This is the source code of our IEEE TCSVT 2019 paper "Bridge-GAN: Interpretable Representation Learning for Text-to-image Synthesis". Please cite the following papers if you use our code.

Mingkuan Yuan and Yuxin Peng, "Bridge-GAN: Interpretable Representation Learning for Text-to-image Synthesis", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), DOI:10.1109/TCSVT.2019.2953753, Nov. 2019. [[pdf]](http://59.108.48.34/tiki/download_paper.php?fileId=201922)

## Training Environment
CUDA 9.0

Python 3.6.8

TensorFlow 1.10.0

## Preparation
Download the preprocessed char-CNN-RNN text embeddings and filename lists for [birds](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE), which should be saved in data/cub/

Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data and extract them to data/cub/images/

Download the [Inception score](https://github.com/hanzhanggit/StackGAN-inception-model) model to evaluation/models/ for evaluating the trained model

Run the following command:

    - sh data_preprocess.sh

## Training
    - run sh train_all.sh to train the model
    
### Trained Model
Download our [trained model](https://drive.google.com/open?id=1XD53s2SfJK8KRTSYBA4uQciPQzFZLOLl) to code/results/00000-bgan-cub-cond-2gpu/ for evaluation
    
## Inception Score Environment
CUDA 8.0

Python 2.7.12

TensorFlow 1.2.1

## Evaluation
    - run sh test_all.sh to evaluate the final inception score
    
## Our Related Work
If you are interested in text-to-image synthesis, you can check our recently published papers about it:

Mingkuan Yuan and Yuxin Peng, "CKD: Cross-task Knowledge Distillation for Text-to-image Synthesis", IEEE Transactions on Multimedia (TMM), DOI:10.1109/TMM.2019.2951463, Nov. 2019. [[pdf]](http://59.108.48.34/tiki/download_paper.php?fileId=201920)

Mingkuan Yuan and Yuxin Peng, "Text-to-image Synthesis via Symmetrical Distillation Networks", 26th ACM Multimedia Conference (ACM MM), pp. 1407-1415, Seoul, Korea, Oct. 22-26, 2018. [[pdf]](http://59.108.48.34/tiki/download_paper.php?fileId=201820)

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.