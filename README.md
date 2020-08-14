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
  and get better average accuracy. What's more, we implement a simple baseline(ResNet-FC)
   and the performance is acceptable.
  

## License
This project is released under [Apache 2.0 license](https://github.com/Media-Smart/vedastr/blob/master/LICENSE).

## Benchmark and model zoo
Note: 
- We test our model on [IIIT5K_3000](http://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset),
 [SVT](http://vision.ucsd.edu/~kai/svt/),
  [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions), 
  [IC13_1015](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads),
[IC15_2077](https://rrc.cvc.uab.es/?ch=4&com=downloads), SVTP,
[CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html).  The training data we used is [MJSynth(MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and
 [SynthText(ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/). You can find the 
 datasets below.
  
| MODEL|CASE SENSITIVE| IIIT5k_3000|	SVT	|IC03_867|	IC13_1015|	 IC15_2077|	SVTP|	CUTE80| AVERAGE|
|:----:|:----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: | :----:|
|[Rosetta]()| False|84.50 | 84.7 | 92.39 | 89.36|68.47|71.32|67.6|80.65|
|[Rosetta-rotate]()| False|84.90 | 84.54 | 91.00 | 87.39|68.94|73.33|69.34|80.72|
|[x] [ResNet-FC]()| False|85.03 | 86.4 | 94| 91.03|70.29|77.67|71.43|82.38|
|[TPS-ResNet-BiLSTM-Attention]()| False|87.87 | 87.02 | 94.12| 91.63|74.92|79.22|75.96|84.86|
|[x] [Small-SATRN]()| False|88.87 | 88.87 | 96.19 | 93.99|79.08|84.81|84.67|87.55|

TPS : [Spatial transformer network](https://arxiv.org/abs/1603.03915)

Small-SATRN: [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396), 
training phase is case sensitive while testing phase is case insensitive.

Rosetta: [Rosetta: Large scale system for text detection and recognition in images](https://arxiv.org/abs/1910.05085).

AVERAGE : Average accuracy over all test datasets

CASE SENSITIVE : If true, the output is case sensitive and contain common characters.
If false, the output is not case sentive and contains only numbers and letters. 

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

Modify some configuration accordingly in the config file like `configs/tps_resnet_bilstm_attn.py`

b. Run

```shell
python tools/train.py configs/tps_resnet_bilstm_attn.py 
```

Snapshots and logs will be generated at `vedastr/workdir` by default.

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/tps_resnet_bilstm_attn.py `

b. Run

```shell
python tools/test.py configs/tps_resnet_bilstm_attn.py path_to_tps_resnet_bilstm_attn_weights
```

## Inference
a. Run

```shell
python tools/inference.py config-path weight-path img-path
```

## Deploy
a. Install [volksdep](https://github.com/Media-Smart/volksdep) following the 
[official instructions](https://github.com/Media-Smart/volksdep#installation)

b. Benchmark (optional)
```python
python tools/deploy/benchmark.py configs/rosetta.py checkpoint_path image_file_path --calibration_images image_path

```
More available arguments are detailed in [tools/deploy/benchmark.py](https://github.com/Media-Smart/vedastr/blob/master/tools/deploy/benchmark.py).

The result of rosetta is as follows（test device: GTX 1080Ti, test dataset: SVTP）:

| framework  |  version   |     input shape      |         data type         |   throughput(FPS)    |   latency(ms)   |       accuracy       |
|   :---:    |   :---:    |        :---:         |           :---:           |        :---:         |      :---:      |        :---:         |
|  pytorch   |   1.5.1    |   (1, 1, 32, 100)    |           fp32            |         160          |      6.16       | acc: 0.7194, edit_distance: 0.8936 |
|  tensorrt  |  7.1.3.4   |   (1, 1, 32, 100)    |           fp32            |         390          |      2.57       | acc: 0.7194, edit_distance: 0.8936 |
|  pytorch   |   1.5.1    |   (1, 1, 32, 100)    |           fp16            |         144          |      6.48       | acc: 0.7178, edit_distance: 0.8934 |
|  tensorrt  |  7.1.3.4   |   (1, 1, 32, 100)    |           fp16            |         377          |       2.6       | acc: 0.7194, edit_distance: 0.8936 |
|  tensorrt  |  7.1.3.4   |   (1, 1, 32, 100)    |       int8(entropy)       |         640          |      1.65       | acc: 0.7178, edit_distance: 0.8944 |
|  tensorrt  |  7.1.3.4   |   (1, 1, 32, 100)    |      int8(entropy_2)      |         607          |      1.75       | acc: 0.7194, edit_distance: 0.8943 |
|  tensorrt  |  7.1.3.4   |   (1, 1, 32, 100)    |       int8(minmax)        |         606          |      1.72       | acc: 0.7209, edit_distance: 0.8948 |


c. Export model as ONNX or TensorRT engine format

```python
python tools/deploy/export.py configs/rosetta.py checkpoint_path image_file_path out_model_path
```

  More available arguments are detailed in [tools/deploy/export.py](https://github.com/Media-Smart/vedastr/blob/master/tools/deploy/export.py).

d. Inference SDK

  You can refer to [FlexInfer](https://github.com/Media-Smart/flexinfer) for details.

## Contact

This repository is currently maintained by  Jun Sun([@ChaseMonsterAway](https://github.com/ChaseMonsterAway)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) , [mmdetection](https://github.com/open-mmlab/mmdetection), [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) and [vedaseg](https://github.com/Media-Smart/vedaseg) thanks to [open-mmlab](https://github.com/open-mmlab), [clovaai](https://github.com/clovaai), [Media-Smart](https://github.com/Media-Smart).

