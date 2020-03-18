## Introduction
vedastr is an open source scene text recognition toolbox based on PyTorch.

## Features
- Modular Design\
  \
  This project is highly modularized, you can easily implement a customized scene text recognition model
   by combining different modules like toy brick. Besides, you can create a new module because our system
   can extend easily. \
  We mainly decompose current model into two parts, the **body** and the **head**. 
  - We decompose body into sub modules, such as **feature extraction module**, **rectification module**,
   **sequence encoder module** and **collection module**. The collection module gives infinite 
   possibilities to transfer features to different module of freedom. You can replace or change
    component arbitrarily, e.g., switching from ResNet to VGG in feature extraction module,
     using rectification module or not, using different encoder scheme such as RNN or CNN. 
     Even more, you can create a new module easily.
   - We implement different head, e.g., attention head and fully connection head. You can switch from
    attention head to fully connection head easily by changing few lines of config file. 
    You can create a new head embedded in the current system. 
 
 Also, you can configure any other modules in scene text recognition system, e.g., datasets, dataloader,
  transformer, converter, optimizer and learning rate scheduler.  
   
    

- Support of current classical frameworks\
  \
  The toolbox supports several popular scene text recognition framework, e.g., [CRNN](https://arxiv.org/abs/1507.05717),
   [TPS-ResNet-BiLSTM-Attention](https://github.com/clovaai/deep-text-recognition-benchmark), etc.

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/vedastr/blob/master/LICENSE).

## Benchmark and model zoo
Note: 
- We test our model on [IIIT5K_3000](http://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset),
 [SVT](http://vision.ucsd.edu/~kai/svt/),
  [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions), 
  [IC13_1015](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads),
[IC15_2077](https://rrc.cvc.uab.es/?ch=4&com=downloads),SVTP,
[CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html).  The training data we used is [MJSynth(MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and
 [SynthText(ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/). You can find the 
 datasets below.
  
| |CASE SENSITIVE| IIIT5k_3000|	SVT	|IC03_867|	IC13_1015|	 IC15_2077|	SVTP|	CUTE80| AVERAGE|MODEL|
|:----:|:----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: | :----:|:----:|
|TPS-ResNet-BiLSTM-Attention| False|87.33 | 87.79 | 95.04| 92.61|74.45|81.09|74.91|84.95|[TPS-ResNet-BiLSTM-Attention](https://drive.google.com/open?id=1Gr7UwSBrkmN0Ldgfbll3mdgSdI_k5o6O)|

AVERAGE : Average accuracy over all test datasets\
TPS : [Spatial transformer network](https://arxiv.org/abs/1603.03915)\
CASE SENSITIVE : If true, the output is case sensitive and contain common characters.
If false, the output is not case sentive and contains only numbers and letters. 


## Installation
### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 9.0
- Python 3.6.9

### Install vedastr

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedastr python=3.6 -y
conda activate vedastr
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/),
 *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedastr repository.

```shell
git clone https://github.com/Media-Smart/vedastr.git
cd vedastr
vedastr_root=${PWD}
```

d. Install dependencies.

```shell
pip install -r requirements.txt
```

## Prepare data
a. Download Lmdb data from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark),
 which contains training data, validation data and evaluation data. 

b. Make directory data as follows:

```shell
cd ${vedastr_root}
mkdir ${vedastr_root}/data
```

c. Put the download Lmdb data into this data directory, the structure of data directory will look like as follows: 

```shell
data
└── data_lmdb_release
    ├── evaluation
    ├── training
    │   ├── MJ
    │   │   ├── MJ_test
    │   │   ├── MJ_train
    │   │   └── MJ_valid
    │   └── ST
    └── validation
```



## Train

a. Config

Modify some configuration accordingly in the config file like `configs/clova.py`

b. Run

```shell
python tools/trainval.py configs/clova.py
```

Snapshots and logs will be generated at `vedastr/workdir`.

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/clova.py`

b. Run

```shell
python tools/test.py configs/clova.py path_to_clova_weights
```

## Contact

This repository is currently maintained by  Jun Sun([@ChaseMonsterAway](https://github.com/ChaseMonsterAway)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) , [mmdetection](https://github.com/open-mmlab/mmdetection), [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) and [vedaseg](https://github.com/Media-Smart/vedaseg) thanks to [open-mmlab](https://github.com/open-mmlab), [clovaai](https://github.com/clovaai), [Media-Smart](https://github.com/Media-Smart).

