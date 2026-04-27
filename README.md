# ScaLR+

PyTorch code and models for ScaLR+, an improved version of ScaLR (CVPR24) used in IGLOSS.

ScaLR+ provides high-quality 3D self-supervised features on lidar data. These features are obtained by distilling (without any annotations) DINOv2 visual features into high-capacity 3D backbones using a mixture of diverse autonomous driving datasets. 

[**Three Pillars improving Vision Foundation Model Distillation for Lidar**](https://arxiv.org/abs/2310.17504)\
[*Gilles Puy*<sup>1</sup>](https://sites.google.com/site/puygilles/home),
[*Spyros Gidaris*<sup>1</sup>](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en),
[*Alexandre Boulch*<sup>1</sup>](http://boulch.eu),
[*Oriane Siméoni*<sup>1</sup>](https://osimeoni.github.io/),
[*Corentin Sautier*<sup>1,3</sup>](https://csautier.github.io/),
[*Patrick Pérez*<sup>2,*</sup>](https://ptrckprz.github.io/),
[*Andrei Bursuc*<sup>1</sup>](https://abursuc.github.io/),
[*Renaud Marlet*<sup>1,3</sup>](http://imagine.enpc.fr/~marletr/)  
<sup>1</sup>*valeo.ai, France*.\
<sup>2</sup>*Kyutai, France*.\
<sup>3</sup>*LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France*.\
<sup>*</sup>*Work done at valeo.ai.*

[**IGLOSS: Image Generation for Lidar Open-vocabulary Semantic Segmentation**](https://arxiv.org/abs/2604.01361)\
[*Nermin Samet*<sup>1</sup>](https://nerminsamet.github.io/),
[*Gilles Puy*<sup>1</sup>](https://sites.google.com/site/puygilles/home),
[*Renaud Marlet*<sup>1,2</sup>](http://imagine.enpc.fr/~marletr/)  
<sup>1</sup>*valeo.ai, France*.\
<sup>2</sup>*LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France*.


If you find this code or work useful, please cite the following papers:
```
@inproceedings{scalr,
  title={Three Pillars improving Vision Foundation Model Distillation for Lidar},
  author={Puy, Gilles and Gidaris, Spyros and Boulch, Alexandre and Sim\'eoni, Oriane and Sautier, Corentin and P\'erez, Patrick and Bursuc, Andrei and Marlet, Renaud},
  booktitle={CVPR},
  year={2024},
}

@inproceedings{igloss,
  author = {Nermin Samet and Gilles Puy and Renaud Marlet},
  title = {IGLOSS: Image Generation for Lidar Open-vocabulary Semantic Segmentation},
  booktitle = {arXiv},
  year = {2026},
}
```

## Overview

- [ScaLR+ vs ScaLR](#scalr-vs-scalr)
- [Installation](#installation)
- [Available models](#available-models)
- [Evaluation](#evaluation)
- [Training](#training)

## ScaLR+ vs ScaLR


### Change of training recipe

The following changes were made in ScaLR+ compared to ScaLR:
- Use stochastic depth during distillation;
- Use GeLU instead of ReLU in the WaffleIron backbone;
- Use a MLP distillation head instead of a linear one;
- Use higher resolution images during distillation;
- Distill for more epochs.

The pretrained ScaLR+ backbone is available [here](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-pretrained.tar.gz).

### Improvement in linear probing

|  Dataset      | ScaLR (mIoU) | ScaLR+ (mIoU) |
| ------------- |:------------:|:-------------:|
| nuScenes      |    67.8 %    |  [**70.4 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-nuscenes.tar.gz)   |
| SemanticKITTI |    55.8 %    |  [**61.0 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-kitti.tar.gz)   |
| Pandar 64     |    37.9 %    |  [**43.5 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-panda64.tar.gz)   |
| Pandar GT     |    34.5 %    |  [**40.7 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-pandagt.tar.gz)   |

### Improvement in finetuning

|  Dataset      | Split | ScaLR (mIoU) | ScaLR+ (mIoU) |
| ------------- | -----:|:------------:|:-------------:|
| nuScenes      | 1 %   |    50.7 %    |  [**54.1 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-nuscenes-1p.tar.gz)   |
|               | 10 %  |    69.2 %    |  [**70.7 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-nuscenes-10p.tar.gz)   |
|               | 100 % |  **78.4 %**  |  [**78.4 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-nuscenes-100p.tar.gz)   |
| SemanticKITTI | 1 %   |    55.8 %    |  [**57.4 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-kitti-1p.tar.gz)   |
|               | 100 % |  **65.8 %**  |    [65.2 %](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-kitti-100p.tar.gz)     |
| Pandar 64     | 100 % |    48.3 %    |  [**50.7 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-panda64-100p.tar.gz)   |
| Pandar GT     | 100 % |    41.1 %    |  [**44.0 %**](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-pandagt-100p.tar.gz)   |


## Installation

### Environment

```bash
conda create -n scalr_plus
conda activate scalr_plus
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch
pip install pyaml==25.7.0 tqdm==4.67.1 scipy==1.15.3 tensorboard==2.20.0 nuscenes-devkit==1.2.0 pandas==2.3.1 transforms3d==0.4.2 numpy==1.26.0 timm==1.0.26
git clone -b scalr_plus https://github.com/valeoai/ScaLR
cd ScaLR
```

Download and untar the following [file](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/info_datasets.tar.gz):
```bash
wget https://github.com/valeoai/ScaLR/releases/download/v0.2.0/info_datasets.tar.gz
tar -xvzf info_datasets.tar.gz
rm info_datasets.tar.gz
```


### Datasets

We use the following datasets: [nuScenes](https://www.nuscenes.org/nuscenes), [SemanticKITTI](https://www.semantic-kitti.org/) and [PandaSet](https://pandaset.org/).

Please download them under the same root directory. The folder structure must be:
```
/path/to/datasets/
|
|- nuscenes/
|  |- lidarseg/
|  | ...
|  |- v1.0-trainval
|
|- semantic_kitti/
|  |- calib/
|  | ...
|  |- dataset/
|
|- pandaset/
|  |- 001/
|  | ...
|  |- 124/
```


## Available models

### Pretrained model with no annotation

We provide the following model, pretrained by distillation and without using any annotations. It can be used, e.g., for unsupervised tasks.

| WaffleIron | Distilled from  | Training datasets                           | Link |
|------------|-----------------|---------------------------------------------|------|
| WI-48-768  | DINOv2 ViT-L/14 | nuScenes & SemKITTI & Pandar 64 & Pandar GT | [backbone + dist. head](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-pretrained.tar.gz) |


### Model adapted to downstream tasks

#### Linear probing

We provide here models obtained after *linear probing* the ScaLR+ pretrained [backbone](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-pretrained.tar.gz).

| WaffleIron | Distilled from  | Linearly probed on |  mIoU  | Link |
| ---------- | --------------- | ------------------ |:------:|------|
| WI-48-768  | DINOv2 ViT-L/14 | nuScenes           | 70.4 % | [backbone + class. head](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-nuscenes.tar.gz) |
| WI-48-768  | DINOv2 ViT-L/14 | SemanticKITTI      | 61.0 % | [backbone + class. head](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-kitti.tar.gz)    |
| WI-48-768  | DINOv2 ViT-L/14 | Pandar 64          | 43.5 % | [backbone + class. head](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-panda64.tar.gz)  |
| WI-48-768  | DINOv2 ViT-L/14 | Pandar GT          | 40.7 % | [backbone + class. head](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-pandagt.tar.gz)  |


#### Finetuning on complete datasets

We provide here models obtained after finetuning the ScaLR+ pretrained [backbone](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-pretrained.tar.gz) on the full datasets of nuScenes, SemanticKITTI, Pandar 64 or Pandar GT.

| WaffleIron | Distilled from  | Finetuned on  | mIoU  | Link |
|------------|-----------------|---------------|:-----:|------|
| WI-48-768  | DINOv2 ViT-L/14 | nuScenes      | 78.4 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-nuscenes-100p.tar.gz) |
| WI-48-768  | DINOv2 ViT-L/14 | SemanticKITTI | 65.2 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-kitti-100p.tar.gz)    |
| WI-48-768  | DINOv2 ViT-L/14 | Pandar 64     | 50.7 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-panda64-100p.tar.gz)  |
| WI-48-768  | DINOv2 ViT-L/14 | Pandar GT     | 44.0 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-pandagt-100p.tar.gz)  |


#### Few-shot learning

We provide here models obtained after finetuning the ScaLR+ pretrained [backbone](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-pretrained.tar.gz) on subsets of nuScenes or SemanticKITTI.

| WaffleIron | Distilled from  | Finetuned on      | mIoU  | Link |
|------------|-----------------|-------------------|:-----:|-----|
| WI-48-768  | DINOv2 ViT-L/14 | 1 % nuScenes      | 54.1 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-nuscenes-1p.tar.gz)  |
| WI-48-768  | DINOv2 ViT-L/14 | 10 % nuScenes     | 70.7 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-nuscenes-10p.tar.gz) |
| WI-48-768  | DINOv2 ViT-L/14 | 1 % SemanticKITTI | 57.4 % | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-finetuning-kitti-1p.tar.gz)     |


#### Downloading the models

For any of the model above, please download the associated file and untar it in the working directory `ScaLR/`. For example:
```bash
wget https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-linear_probing-nuscenes.tar.gz
tar -xvzf WI_768-ScaLR_plus-linear_probing-nuscenes.tar.gz
rm WI_768-ScaLR_plus-linear_probing-nuscenes.tar.gz
```

## Evaluation

We explain here how to evaluate our models.

### Dataset setups

First, please set the following environment variable so that it points to the root directory where you stored your datasets.
```bash
PATH_TO_DATASETS=/path/to/datasets/
```

Then, please use one of the following command line to set the evaluation dataset MACROs for `NuScenes`, `SemanticKITTI`, `Pandar 64` or `Pandar GT`:
```bash
# NuScenes
DATASET_NAME=nuscenes; DATASET_PATH=nuscenes;
```
```bash
# SemanticKITTI
DATASET_NAME=semantic_kitti; DATASET_PATH=semantic_kitti;
```
```bash
# Pandar 64
DATASET_NAME=panda64; DATASET_PATH=pandaset;
```
```bash
# Pandar GT
DATASET_NAME=pandagt; DATASET_PATH=pandaset;
```

### Linear probing evaluation

In order to evaluate the linear probing performance of our models, please use the following command:
```bash
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_linprob.yaml \
--log_path logs/linear_probing/WI_768-ScaLR_plus/$DATASET_NAME/ \
--linprob \
--gpu 0 \
--restart \
--eval
```

If needed, for evaluation, you can reduce the batch size and number of workers in `configs/downstream/$DATASET_PATH/WI_768_linprob.yaml`.


### Finetuning evaluation

In order to evaluate the performance of our provided finetuned models, please use the following command:
```bash
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_100p.yaml \
--log_path logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/100p/ \
--gpu 0 \
--restart \
--eval
```


### Few-shot learning evaluation

In order to evaluate the performance of our provided models finetuned on subsets of nuScenes or SemanticKITTI, please use the following commands:

```bash
# 1 % of nuScenes
SPLIT=1p
DATASET_NAME=nuscenes

python finetune.py \
--dataset ${DATASET_NAME}_${SPLIT} \
--path_dataset $PATH_TO_DATASETS/$DATASET_NAME/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_$SPLIT.yaml \
--log_path logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/$SPLIT/ \
--gpu 0 \
--restart \
--eval


# 10 % of nuScenes
SPLIT=10p
DATASET_NAME=nuscenes

python finetune.py \
--dataset ${DATASET_NAME}_${SPLIT} \
--path_dataset $PATH_TO_DATASETS/$DATASET_NAME/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_$SPLIT.yaml \
--log_path logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/$SPLIT/ \
--gpu 0 \
--restart \
--eval


# 1 % of SemanticKITTI
SPLIT=1p
DATASET_NAME=semantic_kitti 

python finetune.py \
--dataset ${DATASET_NAME}_${SPLIT} \
--path_dataset $PATH_TO_DATASETS/$DATASET_NAME/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_$SPLIT.yaml \
--log_path logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/$SPLIT/ \
--gpu 0 \
--restart \
--eval
```


## Training

### ScaLR pretraining by distillation

Please set the following environment variable so that it points to the root directory where you stored your datasets.
```bash
PATH_TO_DATASETS=/path/to/datasets/
```

The distillation can then be launched as follows.
```bash
python distill.py \
--dataset merged_datasets \
--path_dataset $PATH_TO_DATASETS/ \
--log_path my_own_logs/pretraining/WI_768-ScaLR_plus/ \
--config configs/pretrain/WI_768-ScaLR_plus.yaml \
--multiprocessing-distributed
```
The new distilled model will be saved in the folder `./my_own_logs/`

### Downstream trainings

We now provide the command lines that can be used for linear probing or finetuning a distilled model. 

**In the examples below, we start from our distilled model available** [here](https://github.com/valeoai/ScaLR/releases/download/v0.2.0/WI_768-ScaLR_plus-pretrained.tar.gz).

For any of the experiments below, you must specify the dataset used for downstream linear probing or finetuning by setting the variable `DATASET_NAME` and `DATASET_PATH` (see this [section](#dataset-setups)).

#### Linear probing

Use the following command:
```bash
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_linprob.yaml \
--pretrained_ckpt logs/pretrain/WI_768-ScaLR_plus/ckpt_last.pth \
--log_path my_own_logs/linear_probing/WI_768-ScaLR_plus/$DATASET_NAME/ \
--multiprocessing-distributed \
--linprob
```
The model model will be saved in the folder `./my_own_logs/`.

You can relaunch the evaluation on the validation set by adding `--restart --eval` in the above command.

#### Finetuning on the complete training sets

Use the following command:
```bash
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_100p.yaml \
--pretrained_ckpt logs/pretrain/WI_768-ScaLR_plus/ckpt_last.pth \
--log_path my_own_logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/100p/ \
--multiprocessing-distributed
```
The model model will be saved in the folder `./my_own_logs/`.

You can relaunch the evaluation on the validation set by adding `--restart --eval` in the above command.


#### Finetuning on the partial training sets of nuScenes or SemanticKITTI

We now provide the scripts to finetune the models with different percentage of the training datasets.

For finetuning on the split of **1% of nuScenes**, please use:
```bash
SPLIT=1p
DATASET_NAME=nuscenes

python finetune.py \
--dataset ${DATASET_NAME}_${SPLIT} \
--path_dataset $PATH_TO_DATASETS/$DATASET_NAME/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_1p.yaml \
--pretrained_ckpt logs/pretrain/WI_768-ScaLR_plus/ckpt_last.pth \
--log_path my_own_logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/$SPLIT/ \
--multiprocessing-distributed
```

For finetuning on the split of **10% of nuScenes**, please use:
```bash
SPLIT=10p
DATASET_NAME=nuscenes

python finetune.py \
--dataset ${DATASET_NAME}_${SPLIT} \
--path_dataset $PATH_TO_DATASETS/$DATASET_NAME/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_1p.yaml \
--pretrained_ckpt logs/pretrain/WI_768-ScaLR_plus/ckpt_last.pth \
--log_path my_own_logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/$SPLIT/ \
--multiprocessing-distributed
```

For finetuning on the split of **1% of SemanticKITTI**, please use:
```bash
SPLIT=1p
DATASET_NAME=semantic_kitti

python finetune.py \
--dataset ${DATASET_NAME}_${SPLIT} \
--path_dataset $PATH_TO_DATASETS/$DATASET_NAME/ \
--config_pretrain configs/pretrain/WI_768-ScaLR_plus.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_1p.yaml \
--pretrained_ckpt logs/pretrain/WI_768-ScaLR_plus/ckpt_last.pth \
--log_path my_own_logs/finetuning/WI_768-ScaLR_plus/$DATASET_NAME/$SPLIT/ \
--multiprocessing-distributed
```


## Acknowledgements
We thank the authors of
```
@inproceedings{berman18lovasz,
  title = {The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks},
  author = {Berman, Maxim and Triki, Amal Rannen and Blaschko, Matthew B.},
  booktitle = {CVPR},
  year = {2018},
}
```
for making their [implementation](https://github.com/bermanmaxim/LovaszSoftmax) of the Lovász loss publicly available

## License
ScaLR+ is released under the [Apache 2.0 license](./LICENSE).

The implementation of the Lovász loss in `utils/lovasz.py` is released under [MIT Licence](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/LICENSE).
