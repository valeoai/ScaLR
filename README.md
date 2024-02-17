# ScaLR

[**Three Pillars improving Vision Foundation Model Distillation for Lidar**](https://arxiv.org/abs/2310.17504)  
[*Gilles Puy*<sup>1</sup>](https://sites.google.com/site/puygilles/home),
[*Spyros Gidaris*<sup>1</sup>](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en),
[*Alexandre Boulch*<sup>1</sup>](http://boulch.eu),
[*Oriane Siméoni*<sup>1</sup>](https://osimeoni.github.io/), 
[*Corentin Sautier*<sup>1,2</sup>](https://csautier.github.io/), 
[*Patrick Pérez*<sup>1</sup>](https://ptrckprz.github.io/), 
[*Andrei Bursuc*<sup>1</sup>](https://abursuc.github.io/),
[*Renaud Marlet*<sup>1,2</sup>](http://imagine.enpc.fr/~marletr/)  
<sup>1</sup>*valeo.ai, France* and <sup>2</sup>*LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France*.

If you find this code or work useful, please cite the following [paper](https://arxiv.org/abs/2310.17504):
```
@article{puy23scalr,
  title={Three Pillars improving Vision Foundation Model Distillation for Lidar},
  author={Puy, Gilles and Gidaris, Spyros and Boulch, Alexandre and Sim\'eoni, Oriane and Sautier, Corentin and P\'erez, Patrick and Bursuc, Andrei and Marlet, Renaud},
  journal={arXiv:2310.17504},
  year={2023}
}
```


## Environment

Create the following environment and clone this repo:
```
conda create -n scalr
conda activate scalr
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install pyaml tqdm tensorboard nuscenes-devkit pandas transforms3d
git clone https://github.com/valeoai/WaffleIron
pip install -e WaffleIron/
git clone https://github.com/valeoai/ScaLR
cd ScaLR
```

Download and untar the following file:
```
wget https://github.com/valeoai/ScaLR/releases/download/v0.1.0/info_datasets.tar.gz
tar -xvzf info_datasets.tar.gz
rm info_datasets.tar.gz
```


## Datasets

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

We provide the following distilled model.

| WaffleIron | Distilled from  | using a datasets made of the training sets of    | Name                             | Link         |
|------------|-----------------|--------------------------------------------------|----------------------------------|--------------|
| WI-48-768  | DINOv2 ViT-L/14 | nuScenes & SemanticKITTI & Pandar 64 & Pandar GT | WI_768-DINOv2_ViT_L_14-NS_KI_PD  | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-pretrained.tar.gz) |


We provide the following models obtained after linear probing.

| Distilled model                 | linearly probed on 100% of the training set of | mIoU  | Link                                             |
|---------------------------------|------------------------------------------------|:-----:|--------------------------------------------------|
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | nuScenes                                       | 67.8% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz) |
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | SemanticKITTI                                  | 55.8% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-kitti.tar.gz)                                     |
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | Pandar 64                                      | 37.9% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-panda64.tar.gz)                                     |
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | Pandar GT                                      | 34.5% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-pandagt.tar.gz)                                     |


We provide the following models obtained after finetuning:

| Distilled model                 | finetuned on 100% of the training set of | mIoU  | Link                                       |
|---------------------------------|------------------------------------------|:-----:|--------------------------------------------|
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | nuScenes                                 | 78.4% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-finetuning-nuscenes-100p.tar.gz) |
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | SemanticKITTI                            | 65.8% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-finetuning-kitti-100p.tar.gz)     |
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | Pandar 64                                | 48.3% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-finetuning-panda64-100p.tar.gz)     |
| WI_768-DINOv2_ViT_L_14-NS_KI_PD | Pandar GT                                | 41.1% | [Download](https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-finetuning-pandagt-100p.tar.gz)     |


Finally, we provide the WaffleIron WI-48-768 trained on nuscenes **without** pretraining [here](WI_768-no_pretraining-finetuning-nuscenes-100p.tar.gz). It reaches a mIoU of 78.7%.

For any of the model above, download the associated file and untar it in the working directory `ScaLR/`. For example:
```
wget https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz
tar -xvzf WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz
rm WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz
```

## Downstream experiments (linear probing and finetuning)

Set the following environment variable so that it points to the root directory where you stored your datasets.
```
export PATH_TO_DATASETS=/path/to/datasets/
```

To work on the complete training set of nuScenes, use:
```
export DATASET_NAME=nuscenes
export DATASET_PATH=nuscenes
```

To work on the complete training set of SemanticKITTI, use:
```
export DATASET_NAME=semantic_kitti
export DATASET_PATH=semantic_kitti
```

To work on the complete training set of Pandar 64, use:
```
export DATASET_NAME=panda64
export DATASET_PATH=pandaset
```

To work on the complete training set of Pandar GT, use:
```
export DATASET_NAME=pandagt
export DATASET_PATH=pandaset
```

### Linear probing

To evaluate the performance of our provided models:
```
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_linprob.yaml \
--log_path logs/linear_probing/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/ \
--multiprocessing-distributed \
--fp16 \
--linprob \
--restart \
--eval
```

If needed, for evaluation, you can reduce the batch size and number of workers in `configs/downstream/$DATASET_PATH/WI_768_linprob.yaml`.

To re-run the linear probing experiment, starting from our provided distilled model:
```
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_linprob.yaml \
--pretrained_ckpt logs/pretraining/WI_768-DINOv2_ViT_L_14-NS_KI_PD/model.pth \
--log_path my_own_logs/linear_probing/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/ \
--multiprocessing-distributed \
--fp16 \
--linprob
```
The model model will be saved in the folder `./my_own_logs/`.

### Finetuning on the complete training sets (100%)

To evaluate the performance of our provided models:
```
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_100p.yaml \
--log_path logs/finetuning/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/100p/ \
--multiprocessing-distributed \
--fp16 \
--restart \
--eval
```

To re-run the finetuning experiment on the complete training set, starting from our provided distilled model:
```
python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_100p.yaml \
--pretrained_ckpt logs/pretraining/WI_768-DINOv2_ViT_L_14-NS_KI_PD/model.pth \
--log_path my_own_logs/finetuning/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/100p/ \
--multiprocessing-distributed \
--fp16
```
The model model will be saved in the folder `./my_own_logs/`.

### Finetuning on the partial training sets of nuScenes or SemanticKITTI

For finetuning on the split of 1% of nuScenes, use:
```
export DATASET_NAME=nuscenes_1p
export DATASET_PATH=nuscenes

python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_1p.yaml \
--pretrained_ckpt logs/pretraining/WI_768-DINOv2_ViT_L_14-NS_KI_PD/model.pth \
--log_path my_own_logs/finetuning/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/1p/ \
--multiprocessing-distributed \
--fp16
```

For finetuning on the split of 10% of nuScenes, use:
```
export DATASET_NAME=nuscenes_10p
export DATASET_PATH=nuscenes

python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_10p.yaml \
--pretrained_ckpt logs/pretraining/WI_768-DINOv2_ViT_L_14-NS_KI_PD/model.pth \
--log_path my_own_logs/finetuning/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/10p/ \
--multiprocessing-distributed \
--fp16
```

For finetuning on the split of 1% of SemanticKITTI, use:
```
export DATASET_NAME=semantic_kitti_1p
export DATASET_PATH=semantic_kitti

python finetune.py \
--dataset $DATASET_NAME \
--path_dataset $PATH_TO_DATASETS/$DATASET_PATH/ \
--config_pretrain configs/pretrain/WI_768_pretrain.yaml \
--config_downstream configs/downstream/$DATASET_NAME/WI_768_finetune_1p.yaml \
--pretrained_ckpt logs/pretraining/WI_768-DINOv2_ViT_L_14-NS_KI_PD/model.pth \
--log_path my_own_logs/finetuning/WI_768-DINOv2_ViT_L_14-NS_KI_PD/$DATASET_NAME/1p/ \
--multiprocessing-distributed \
--fp16
```

## Pretraining by distillation

Download the DINOv2 ViT-L/14 model:
```
mkdir dinov2_weights
cd dinov2_weights
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
cd ..
```

Set the following environment variable so that it points to the root directory where you stored your datasets.
```
export PATH_TO_DATASETS=/path/to/datasets/
```

Start the distillation as follows:
```
python distill.py \
--dataset merged_datasets \
--path_dataset $PATH_TO_DATASETS/ \
--log_path my_own_logs/pretraining/WI_768-DINOv2_ViT_L_14-NS_KI_PD/ \
--config configs/pretrain/WI_768_pretrain.yaml \
--fp16 \
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
for making their [implementation](https://github.com/bermanmaxim/LovaszSoftmax) of the Lovász loss publicly available, and the authors of 
```
@article{oquab2024dinov,
  title={{DINO}v2: Learning Robust Visual Features without Supervision},
  author={Maxime Oquab and Timoth{\'e}e Darcet and Th{\'e}o Moutakanni and Huy V. Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel HAZIZA and Francisco Massa and Alaaeldin El-Nouby and Mido Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Herve Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
  journal={TMLR},
  year={2024},
}
```
for making their [code](https://github.com/facebookresearch/dinov2) and model publicly available.

## License
ScaLR is released under the [Apache 2.0 license](./LICENSE). 

The implementation of the Lovász loss in `utils/lovasz.py` is released under [MIT Licence](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/LICENSE).

The implementation of DINOv2 (`models/dinov2/` and `models/dinov2_vision_transformer.py`) is released under the [Apache 2.0 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).