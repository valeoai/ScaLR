# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import yaml
import torch
import random
import warnings
import argparse
import numpy as np
import utils.transforms as tr
from waffleiron import Segmenter
from utils.metrics import SemSegLoss
from utils.finetuner import Finetuner
from utils.scheduler import WarmupCosine
from datasets import LIST_DATASETS, Collate


def param_groups_lrd(
    model,
    weight_decay=0.05,
    no_weight_decay_list=[],
    layer_decay=0.75,
    no_wdecay_skip=False,
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.waffleiron.channel_mix) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if (no_wdecay_skip is False) and (p.ndim == 1 or n in no_weight_decay_list):
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_waffleiron(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_waffleiron(name, num_layers):
    """
    Assign a parameter with its layer id
    Similar to BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name.startswith("embed"):
        return 0
    elif name.startswith("waffleiron.channel_mix"):
        layer_id = int(name.split(".")[2]) + 1
        return layer_id
    elif name.startswith("waffleiron.spatial_mix"):
        layer_id = int(name.split(".")[2]) + 1
        return layer_id
    else:
        return num_layers


def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_train_augmentations(config):

    list_of_transf = []

    # Two transformations shared across all datasets
    list_of_transf.append(
        tr.LimitNumPoints(
            dims=(0, 1, 2),
            max_point=config["dataloader"]["max_points"],
            random=True,
        )
    )

    # Optional augmentations
    for aug_name in config["augmentations"].keys():
        if aug_name == "rotation":
            for d in config["augmentations"]["rotation"][0]:
                list_of_transf.append(tr.Rotation(inplace=True, dim=d))
        elif aug_name == "flip_xy":
            list_of_transf.append(tr.RandomApply(tr.FlipXY(inplace=True), prob=2 / 3))
        elif aug_name == "scale":
            dims = config["augmentations"]["scale"][0]
            scale = config["augmentations"]["scale"][1]
            list_of_transf.append(tr.Scale(inplace=True, dims=dims, range=scale))
        else:
            raise ValueError(f"Unknown transformation: {aug_name}.")

    print("List of transformations:", list_of_transf)

    return tr.Compose(list_of_transf)


def get_datasets(config, args):

    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["embedding"]["input_feat"],
        "voxel_size": config["embedding"]["voxel_size"],
        "num_neighbors": config["embedding"]["neighbors"],
        "dim_proj": config["waffleiron"]["dim_proj"],
        "grids_shape": config["waffleiron"]["grids_size"],
        "fov_xyz": config["waffleiron"]["fov_xyz"],
    }

    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Train dataset
    train_dataset = DATASET(
        phase="train",
        train_augmentations=get_train_augmentations(config),
        **kwargs,
    )

    # Validation dataset
    val_dataset = DATASET(
        phase="val",
        **kwargs,
    )

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args):

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=Collate(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=Collate(),
    )

    return train_loader, val_loader, train_sampler


def get_optimizer(parameters, config):
    return torch.optim.AdamW(
        parameters,
        lr=config["optim"]["lr"],
        weight_decay=config["optim"]["weight_decay"],
    )


def get_scheduler(optimizer, config, len_train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            config["scheduler"]["epoch_warmup"] * len_train_loader,
            config["scheduler"]["max_epoch"] * len_train_loader,
            config["scheduler"]["min_lr"] / config["optim"]["lr"],
        ),
    )
    return scheduler


def distributed_training(gpu, ngpus_per_node, args, config):

    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # --- Build network
    model = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"]["drop_path"],
        layer_norm=config["waffleiron"]["layernorm"],
    )
    if args.pretrained_ckpt != "":
        # Load pretrained model
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
        if ckpt.get("model_points") is not None:
            ckpt = ckpt["model_points"]
        else:
            ckpt = ckpt["model_point"]
        new_ckpt = {}
        for k in ckpt.keys():
            if k.startswith("module"):
                new_ckpt[k[len("module.") :]] = ckpt[k]
            else:
                new_ckpt[k] = ckpt[k]
        model.classif = torch.nn.Conv1d(
            config["waffleiron"]["nb_channels"], config["waffleiron"]["pretrain_dim"], 1
        )
        model.load_state_dict(new_ckpt)

    # Re-init. classification layer (always a learnable layer)
    classif = torch.nn.Conv1d(
        config["waffleiron"]["nb_channels"], config["classif"]["nb_class"], 1
    )
    torch.nn.init.constant_(classif.bias, 0)
    torch.nn.init.constant_(classif.weight, 0)
    model.classif = torch.nn.Sequential(
        torch.nn.BatchNorm1d(config["waffleiron"]["nb_channels"]),
        classif,
    )

    # For linear probing:
    # We freeze parameters of backbone, except classification layer
    # eval / train mode for batch norm is handled in Finetuner
    if args.linprob:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classif.parameters():
            p.requires_grad = True

    # ---
    args.batch_size = config["dataloader"]["batch_size"]
    args.workers = config["dataloader"]["num_workers"]
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(config["dataloader"]["batch_size"] / ngpus_per_node)
        args.workers = int(
            (config["dataloader"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        # Training on one GPU
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    if args.gpu == 0 or args.gpu is None:
        print(f"Model:\n{model}")
        nb_param = sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6
        print(f"{nb_param} x 10^6 parameters")

    # --- Optimizer
    if config["optim"]["layer_decay"] is not None:
        model_without_ddp = (
            model.module if (args.distributed or args.gpu is None) else model
        )
        print("Apply layer decay")
        param_groups = param_groups_lrd(
            model_without_ddp,
            config["optim"]["weight_decay"],
            layer_decay=config["optim"]["layer_decay"],
        )
        for i in range(len(param_groups)):
            param_groups[i]["lr"] = param_groups[i]["lr_scale"] * config["optim"]["lr"]
        optim = get_optimizer(param_groups, config)
    else:
        optim = get_optimizer(model.parameters(), config)

    # --- Dataset
    train_dataset, val_dataset = get_datasets(config, args)
    train_loader, val_loader, train_sampler = get_dataloader(
        train_dataset, val_dataset, args
    )

    # --- Loss function
    loss = SemSegLoss(
        config["classif"]["nb_class"],
        lovasz_weight=config["loss"]["lovasz"],
    ).cuda(args.gpu)

    # --- Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    scheduler = get_scheduler(optim, config, len(train_loader))

    # --- Training
    mng = Finetuner(
        model,
        loss,
        train_loader,
        val_loader,
        train_sampler,
        optim,
        scheduler,
        config["scheduler"]["max_epoch"],
        args.log_path,
        args.gpu,
        args.world_size,
        args.fp16,
        LIST_DATASETS.get(args.dataset.lower()).CLASS_NAME,
        tensorboard=(not args.eval),
        linear_probing=args.linprob,
    )
    if args.restart:
        mng.load_state()
    if args.eval:
        mng.one_epoch(training=False)
    else:
        mng.train()


def main(args, config):

    # --- Fixed args
    # Device
    args.device = "cuda"
    # Node rank for distributed training
    args.rank = 0
    # Number of nodes for distributed training'
    args.world_size = 1
    # URL used to set up distributed training
    args.dist_url = "tcp://127.0.0.1:4444"
    # Distributed backend'
    args.dist_backend = "nccl"
    # Distributed processing
    args.distributed = args.multiprocessing_distributed

    # Create log directory
    os.makedirs(args.log_path, exist_ok=True)

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Test if use only 1 GPU
    if args.gpu is not None:
        args.gpu = 0
        args.distributed = False
        args.multiprocessing_distributed = False
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism."
        )

    # Multi-GPU or Not
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(
            distributed_training,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, config),
        )
    else:
        # Simply call main_worker function
        distributed_training(args.gpu, ngpus_per_node, args, config)


def get_default_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset",
        default="nuscenes",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/datasets_local/nuscenes/",
    )
    parser.add_argument(
        "--log_path", type=str, required=True, help="Path to log folder"
    )
    parser.add_argument(
        "--restart", action="store_true", default=False, help="Restart training"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="Seed for initializing training"
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to any number to use gpu 0"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--config_pretrain",
        type=str,
        required=True,
        help="Path to config for pretraining",
    )
    parser.add_argument(
        "--config_downstream",
        type=str,
        required=True,
        help="Path to model config downstream",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run validation only",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default="",
        type=str,
        help="Path to pretrained ckpt",
    )
    parser.add_argument(
        "--linprob",
        action="store_true",
        default=False,
        help="Linear probing",
    )

    return parser


if __name__ == "__main__":

    parser = get_default_parser()
    args = parser.parse_args()

    # Load config files
    config = load_model_config(args.config_downstream)
    config_pretrain = load_model_config(args.config_pretrain)

    # Merge config files
    # Embeddings
    config["embedding"] = {}
    config["embedding"]["input_feat"] = config_pretrain["point_backbone"][
        "input_features"
    ]
    config["embedding"]["size_input"] = config_pretrain["point_backbone"]["size_input"]
    config["embedding"]["neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config["embedding"]["voxel_size"] = config_pretrain["point_backbone"]["voxel_size"]
    # Backbone
    config["waffleiron"]["depth"] = config_pretrain["point_backbone"]["depth"]
    config["waffleiron"]["num_neighbors"] = config_pretrain["point_backbone"][
        "num_neighbors"
    ]
    config["waffleiron"]["dim_proj"] = config_pretrain["point_backbone"]["dim_proj"]
    config["waffleiron"]["nb_channels"] = config_pretrain["point_backbone"][
        "nb_channels"
    ]
    config["waffleiron"]["pretrain_dim"] = config_pretrain["point_backbone"]["nb_class"]
    config["waffleiron"]["layernorm"] = config_pretrain["point_backbone"]["layernorm"]

    # For datasets which need larger FOV for finetuning...
    if config["dataloader"].get("new_grid_shape") is not None:
        # ... overwrite config used at pretraining
        config["waffleiron"]["grids_size"] = config["dataloader"]["new_grid_shape"]
    else:
        # ... otherwise keep default value
        config["waffleiron"]["grids_size"] = config_pretrain["point_backbone"][
            "grid_shape"
        ]
    if config["dataloader"].get("new_fov") is not None:
        config["waffleiron"]["fov_xyz"] = config["dataloader"]["new_fov"]
    else:
        config["waffleiron"]["fov_xyz"] = config_pretrain["point_backbone"]["fov"]

    # Launch training
    main(args, config)
