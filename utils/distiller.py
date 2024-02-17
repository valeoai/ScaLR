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


import sys
import torch
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


class Distiller:
    def __init__(
        self,
        model_point,
        model_image,
        loader_train,
        train_sampler,
        optim,
        scheduler,
        max_epoch,
        path,
        rank,
        world_size,
        fp16=True,
        tensorboard=True,
    ):

        # Optim. methods
        self.optim = optim
        self.fp16 = fp16
        self.scaler = GradScaler() if fp16 else None
        self.scheduler = scheduler

        # Dataloaders
        self.max_epoch = max_epoch
        self.loader_train = loader_train
        self.train_sampler = train_sampler

        # Network
        self.model_point = model_point
        self.model_image = model_image
        self.rank = rank
        self.world_size = world_size
        print(f"Trainer on gpu: {self.rank}. World size:{self.world_size}.")

        # Checkpoints
        self.current_epoch = 0
        self.path_to_ckpt = path

        # Monitoring
        if tensorboard and (self.rank == 0 or self.rank is None):
            self.writer_train = SummaryWriter(
                path + "/tensorboard/train/",
                purge_step=self.current_epoch * len(self.loader_train),
                flush_secs=30,
            )
        else:
            self.writer_train = None

    def print_log(self, running_loss):
        if self.rank == 0 or self.rank is None:
            # Global score
            log = f"\nEpoch: {self.current_epoch:d} :\n" + f" Loss = {running_loss:.3f}"
            print(log)
            sys.stdout.flush()

    def gather_scores(self, list_tensors):
        if self.rank == 0:
            tensor_reduced = [
                [torch.empty_like(t) for _ in range(self.world_size)]
                for t in list_tensors
            ]
            for t, t_reduced in zip(list_tensors, tensor_reduced):
                torch.distributed.gather(t, t_reduced)
            tensor_reduced = [sum(t).cpu() for t in tensor_reduced]
            return tensor_reduced
        else:
            for t in list_tensors:
                torch.distributed.gather(t)

    def one_epoch(self):

        # Init.
        loader = self.loader_train
        writer = self.writer_train
        model_image = self.model_image.eval()
        model_point = self.model_point.train()
        if self.rank == 0 or self.rank is None:
            print("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch))
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

        # Log frequency
        print_freq = np.max((len(loader) // 10, 1))

        # Stat.
        running_loss = 0.0

        # Loop over mini-batches
        if self.rank == 0 or self.rank is None:
            bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
            loader = tqdm(loader, bar_format=bar_format)
        for it, batch in enumerate(loader):

            # Extract image features
            images = batch["images"].cuda(self.rank, non_blocking=True)
            with torch.autocast("cuda", enabled=self.fp16):
                feat_im = model_image(images)
                feat_im = feat_im.permute(0, 2, 3, 1)

            # Input to point network
            feat = batch["feat"].cuda(self.rank, non_blocking=True)
            for key in ["pairing_points", "pairing_images"]:
                batch[key] = [p.cuda(self.rank, non_blocking=True) for p in batch[key]]
            cell_ind = batch["cell_ind"].cuda(self.rank, non_blocking=True)
            occupied_cell = batch["occupied_cells"].cuda(self.rank, non_blocking=True)
            neighbors_emb = batch["neighbors_emb"].cuda(self.rank, non_blocking=True)
            net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

            # Get prediction and loss
            with torch.autocast("cuda", enabled=self.fp16):
                feat_point = model_point(*net_inputs)
                # Distillation loss
                k, q = [], []
                for b in range(feat_point.shape[0]):
                    k.append(feat_point[b, :, batch["pairing_points"][b]])
                    m = tuple(batch["pairing_images"][b].T.long())
                    q.append(feat_im[b : b + 1][m])
                k = F.normalize(torch.cat(k, dim=1), p=2, dim=0)
                q = F.normalize(torch.cat(q, dim=0), p=2, dim=1)
                loss = torch.norm(k.transpose(1, 0) - q, dim=1, p=2).mean()
            running_loss += loss.detach()

            # Logs
            if it % print_freq == print_freq - 1 or it == len(loader) - 1:
                # Gather scores
                if self.train_sampler is not None:
                    out = self.gather_scores([running_loss])
                else:
                    out = [running_loss.cpu()]
                if self.rank == 0 or self.rank is None:
                    # Compute scores
                    running_loss_reduced = out[0].item() / self.world_size / (it + 1)
                    # Print score
                    self.print_log(running_loss_reduced)
                    # Save in tensorboard
                    if (writer is not None) and (it == len(loader) - 1):
                        header = "Pretrain"
                        step = self.current_epoch * len(loader) + it
                        writer.add_scalar(header + "/loss", running_loss_reduced, step)
                        writer.add_scalar(
                            header + "/lr", self.optim.param_groups[0]["lr"], step
                        )

            # Gradient step
            self.optim.zero_grad(set_to_none=True)
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                self.optim.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def load_state(self):
        filename = self.path_to_ckpt + "/ckpt_last.pth"
        rank = 0 if self.rank is None else self.rank
        ckpt = torch.load(
            filename,
            map_location=f"cuda:{rank}",
        )
        self.model_point.load_state_dict(ckpt["model_point"])
        if ckpt.get("optim") is None:
            warnings.warn("Optimizer state not available")
        else:
            self.optim.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            if ckpt.get("scheduler") is None:
                warnings.warn("Scheduler state not available")
            else:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.fp16:
            if ckpt.get("scaler") is None:
                warnings.warn("Scaler state not available")
            else:
                self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("epoch") is not None:
            self.current_epoch = ckpt["epoch"] + 1
        print(
            f"Checkpoint loaded on {torch.device(rank)} (cuda:{rank}): {self.path_to_ckpt}"
        )

    def save_state(self):
        if self.rank == 0 or self.rank is None:
            dict_to_save = {
                "epoch": self.current_epoch,
                "model_point": self.model_point.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "scaler": self.scaler.state_dict() if self.fp16 else None,
            }
            filename = self.path_to_ckpt + "/ckpt_last.pth"
            torch.save(dict_to_save, filename)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            self.one_epoch()
            self.save_state()
            self.current_epoch += 1
        if self.rank == 0 or self.rank is None:
            print("Finished Training")
