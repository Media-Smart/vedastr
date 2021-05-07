## Introduction
vedastr is an open source scene text recognition toolbox based on PyTorch. It is designed to be flexible
in order to support rapid implementation and evaluation for scene text recognition task.  

## Features
- **Modular design**\
  We decompose the scene text recognition framework into different components and one can 
  easily construct a customized scene text recognition framework by combining different modules.
  
- **Flexibility**\
  vedastr is flexible enough to be able to easily change the components within a module.

- **Module expansibility**\
  It is easy to integrate a new module into the vedastr project. 

- **Support of multiple frameworks**\
  The toolbox supports several popular scene text recognition framework, e.g., [CRNN](https://arxiv.org/abs/1507.05717),
   [TPS-ResNet-BiLSTM-Attention](https://github.com/clovaai/deep-text-recognition-benchmark), Transformer, etc.

- **Good performance**\
  We re-implement the best model in  [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
  and get better average accuracy. What's more, we devise a new model named [CSTR](https://arxiv.org/abs/2102.10884) which
  achieves nearly SOTA performance.
  

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/vedastr/blob/master/LICENSE).

## Benchmark and model zoo
Note: 
- We use MJSynth(MJ) and SynthText(ST) as training data,  and test the models on 
 [IIIT5K_3000](http://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset),
 [SVT](http://vision.ucsd.edu/~kai/svt/),
  [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions), 
  [IC13_1015](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads),
[IC15_2077](https://rrc.cvc.uab.es/?ch=4&com=downloads), SVTP,
[CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html).
  
| MODEL|CASE SENSITIVE| IIIT5k_3000|	SVT	|IC03_867|	IC13_1015|	 IC15_2077|	SVTP|	CUTE80| AVERAGE|
|:----:|:----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: | :----:|
|[CSTR](https://drive.google.com/file/d/14USWpsW8_HH3BMxYfSWxINlaI1Y26Q1q/view?usp=sharing)| False | 93.7 | 90.1 | 94.8 | 93.2 | 81.6 | 85 | 81.3 | 89.5 |
|[TPS-ResNet-BiLSTM-Attention](https://drive.google.com/file/d/1Zzg1Q8_JTIW4XY-CCmBQhgNkgVsMek-o/view?usp=sharing)| False | 94 | 89.2 | 93.5 | 91.2 | 76.9 | 80.9 | 81.2 | 87.7 |
|[ResNet-CTC](https://drive.google.com/file/d/177FmlOHJWNWgEZwoPlBQBM9mmug_9kue/view?usp=sharing)| False | 91.3 | 85.9 | 90.3 | 88.3 | 70.0 | 74.1 | 73.3 | 83.3 |
|[Small-SATRN]()| False|-|-|-|-|-|-|-|-|

CSTR: [Revisiting Classification Perspective on Scene Text Recognition](https://arxiv.org/abs/2102.10884)

TPS: [Spatial transformer network](https://arxiv.org/abs/1603.03915)

Small-SATRN: [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396), 
training phase is case sensitive while testing phase is case insensitive.

AVERAGE: Average accuracy over all test datasets

CASE SENSITIVE: If true, the output is case sensitive and contain common characters.
If false, the output is not case sensetive and contains only numbers and letters. 

## Installation
### Requirements

- Linux
- Python 3.6+
- PyTorch 1.4.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- Python 3.6.9
- Pytorch: 1.5.1

### Install vedastr

1. Create a conda virtual environment and activate it.

```shell
conda create -n vedastr python=3.6 -y
conda activate vedastr
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/),
 *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

3. Clone the vedastr repository.

```shell
git clone https://github.com/Media-Smart/vedastr.git
cd vedastr
vedastr_root=${PWD}
```

4. Install dependencies.

```shell
pip install -r requirements.txt
```

## Prepare data
1. Download LMDB data from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark),
 which contains training, validation and evaluation data. 
 **Note: we use the ST dataset released by [ASTER](https://github.com/ayumiymk/aster.pytorch#data-preparation).**  

2. Make directory data as follows:

```shell
cd ${vedastr_root}
mkdir ${vedastr_root}/data
```

3. Put the download LMDB data into this data directory, the structure of data directory will look like as follows: 

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

1. Config

Modify some configuration accordingly in the config file like `configs/cstr.py`

2. Training
```shell script
tools/dist_train.sh configs/cstr.py gpu_nums
```

Snapshots and logs will be generated at `vedastr/workdir` by default.

## Test

1. Config

Modify some configuration accordingly in the config file like `configs/cstr.py `

2. Testing
```shell script
tools/dist_test.sh configs/cstr.py checkpoint_path gpu_nums
```

## Inference
1. Run

```shell
python tools/inference.py configs/cstr.py checkpoint_path img_path
```

## Deploy
1. Install [volksdep](https://github.com/Media-Smart/volksdep) following the 
[official instructions](https://github.com/Media-Smart/volksdep#installation)

2. Export model as ONNX

```python
python tools/deploy/export.py configs/resnet_ctc.py checkpoint_path image_file_path out_model_path --onnx
```

  - More available arguments are detailed in [tools/deploy/export.py](https://github.com/Media-Smart/vedastr/blob/master/tools/deploy/export.py).
  - Currently, only `resnet_ctc.py` is supported. 

3. Inference SDK

  You can refer to [FlexInfer](https://github.com/Media-Smart/flexinfer) for details.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{2020vedastr,
    title  = {vedastr: A Toolbox for Scene Text Recognition},
    author = {Sun, Jun and Cai, Hongxiang and Xiong, Yichao},
    url    = {https://github.com/Media-Smart/vedastr},
    year   = {2020}
}
```

## Contact

This repository is currently maintained by Jun Sun([@ChaseMonsterAway](https://github.com/ChaseMonsterAway)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

