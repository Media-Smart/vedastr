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
- We use [MJSynth(MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and
 [SynthText(ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) as training data,  and test the models on 
 [IIIT5K_3000](http://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset),
 [SVT](http://vision.ucsd.edu/~kai/svt/),
  [IC03_867](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions), 
  [IC13_1015](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads),
[IC15_2077](https://rrc.cvc.uab.es/?ch=4&com=downloads), SVTP,
[CUTE80](http://cs-chan.com/downloads_CUTE80_dataset.html). You can find the 
 datasets [below](https://github.com/Media-Smart/vedastr/tree/opencv-version#prepare-data).
  
| MODEL|CASE SENSITIVE| IIIT5k_3000|	SVT	|IC03_867|	IC13_1015|	 IC15_2077|	SVTP|	CUTE80| AVERAGE|
|:----:|:----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: | :----:|
|[ResNet-CTC](https://drive.google.com/file/d/1gtTcc5kpVs_s5a6OR7eBh431Otk_-NrE/view?usp=sharing)| False|87.97 | 84.54 | 90.54 | 88.28 |67.99|72.71|77.08|81.58|
|[ResNet-FC](https://drive.google.com/file/d/1OnUGdv9RFhFbQGXUUkWMcxUZg0mPV0kK/view?usp=sharing)  | False|88.80  | 88.41 | 92.85| 90.34|72.32|79.38|76.74|84.24|
|[TPS-ResNet-BiLSTM-Attention](https://drive.google.com/file/d/1YUOAU7xcrrsAtEqEGtI5ZD7eryP7Zr04/view?usp=sharing)| False|90.93 | 88.72 | 93.89| 92.12|76.41|80.31|79.51|86.49|
|[Small-SATRN](https://drive.google.com/file/d/1bcKtEcYGIOehgPfGi_TqPkvrm6rjOUKR/view?usp=sharing)| False|91.97 | 88.10 | 94.81 | 93.50|75.64|83.88|80.90|87.19|

TPS : [Spatial transformer network](https://arxiv.org/abs/1603.03915)

Small-SATRN: [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396), 
training phase is case sensitive while testing phase is case insensitive.

AVERAGE : Average accuracy over all test datasets

CASE SENSITIVE : If true, the output is case sensitive and contain common characters.
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
1. Download Lmdb data from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark),
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

Modify configuration files in [configs/](configs) according to your needs(e.g. [configs/tps_resnet_bilstm_attn.py](configs/tps_resnet_bilstm_attn.py)). 

2. Run

```shell
# train using GPUs with gpu_id 0, 1, 2, 3
python tools/train.py configs/tps_resnet_bilstm_attn.py "0, 1, 2, 3" 
```

Snapshots and logs by default will be generated at `${vedastr_root}/workdir/name_of_config_file`(you can specify workdir in config files).

## Test

1. Config

Modify configuration as you wish(e.g. [configs/tps_resnet_bilstm_attn.py](configs/tps_resnet_bilstm_attn.py)).

2. Run

```shell
# test using GPUs with gpu_id 0, 1
./tools/dist_test.sh configs/tps_resnet_bilstm_attn.py path/to/checkpoint.pth "0, 1" 
```

## Inference
1. Run

```shell
# inference using GPUs with gpu_id 0
python tools/inference.py configs/tps_resnet_bilstm_attn.py checkpoint_path img_path "0"
```

## Deploy
1. Install [volksdep](https://github.com/Media-Smart/volksdep) following the 
[official instructions](https://github.com/Media-Smart/volksdep#installation)

2. Benchmark (optional)
```python
# Benchmark model using GPU with gpu_id 0
CUDA_VISIBLE_DEVICES="0" python tools/benchmark.py configs/resnet_ctc.py checkpoint_path out_path --dummy_input_shape "3,32,100"
```

More available arguments are detailed in [tools/deploy/benchmark.py](https://github.com/Media-Smart/vedastr/blob/master/tools/deploy/benchmark.py).

The result of resnet_ctc is as follows(test device: Jetson AGX Xavier, CUDA:10.2):

| framework  |  version   |     input shape      |         data type         |   throughput(FPS)    |   latency(ms)   |
|   :---:    |   :---:    |        :---:         |           :---:           |        :---:         |      :---:      |
|  pytorch   |   1.5.0    |   (1, 1, 32, 100)    |           fp32            |          64          |      15.81      |
|  tensorrt  |  7.1.0.16  |   (1, 1, 32, 100)    |           fp32            |         109          |      9.66       |
|  pytorch   |   1.5.0    |   (1, 1, 32, 100)    |           fp16            |         113          |      10.75      |
|  tensorrt  |  7.1.0.16  |   (1, 1, 32, 100)    |           fp16            |         308          |      3.55       |
|  tensorrt  |  7.1.0.16  |   (1, 1, 32, 100)    |      int8(entropy_2)      |         449          |      2.38       |



3. Export model to ONNX format

```python
# export model to onnx using GPU with gpu_id 0
CUDA_VISIBLE_DEVICES="0" python tools/torch2onnx.py configs/resnet_ctc.py checkpoint_path --dummy_input_shape "3,32,100" --dynamic_shape
```

  More available arguments are detailed in [tools/torch2onnx.py](https://github.com/Media-Smart/vedastr/blob/master/tools/torch2onnx.py).

4. Inference SDK

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

