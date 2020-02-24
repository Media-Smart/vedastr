## Introduction
vedastr is an open source scene text recognition toolbox based on PyTorch.

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

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

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
a. Download Lmdb data from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark), which contains training data, validation data and evaluation data. 

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

