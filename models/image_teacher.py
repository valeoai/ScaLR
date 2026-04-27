# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
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

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T


class Preprocessing:
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self):
        super().__init__()
        self.preprocessing_img = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, image):
        return self.preprocessing_img(image)


class ImageTeacher(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ImageNet RGB normalization
        self.preprocessing = Preprocessing()

        # ViT
        assert config["image_backbone"]["images_encoder"].startswith("timm_")
        model_name = config["image_backbone"]["images_encoder"][len("timm_") :]
        assert model_name in timm.list_models(pretrained=True)
        height, width = config["image_backbone"]["im_size"]
        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            img_size=(height, width),
        )
        patch_size = self.encoder.patch_embed.patch_size
        assert patch_size[0] == patch_size[1]
        patch_size = patch_size[0]
        embed_dim = self.encoder.embed_dim

        # Update preprocessing
        data_config = timm.data.resolve_model_data_config(self.encoder)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        self.preprocessing.preprocessing_img = T.Normalize(
            mean=transforms.transforms[-1].mean,
            std=transforms.transforms[-1].std,
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.which_feature = config["image_backbone"]["feat"]
        print("Image teacher:")
        print(f"==> model_name: {model_name}")
        print(f"==> patch_size: {patch_size}")
        print(f"==> embed_dim: {embed_dim}")

        # Compute feature size
        height, width = config["image_backbone"]["im_size"]
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        self.f_height = height // self.patch_size
        self.f_width = width // self.patch_size

        # Create decoder - Just upsampling in our case
        self.decoder = nn.Upsample(
            scale_factor=patch_size, mode="bilinear", align_corners=True
        )

        # Teacher must stay frozen
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()

    def train(self, mode):
        if mode:
            raise ValueError("Image teacher cannot be set in train mode")
        return super().train(mode)

    def forward(self, x):
        # Check that teacher is in eval mode
        assert (not self.encoder.training) and (not self.decoder.training)

        # Go through frozen encoder
        with torch.no_grad():
            x = self.preprocessing(x)

            batch_size = x.shape[0]
            assert self.which_feature in ["x", "x_pre_norm"]
            x = self.encoder.forward_intermediates(
                x,
                1,
                return_prefix_tokens=False,
                norm=True if self.which_feature == "x" else False,
                stop_early=True,
                intermediates_only=True,
            )[0]
            assert x.shape == (batch_size, self.embed_dim, self.f_height, self.f_width)

            # Go through decoder
            temp_x = []
            for id_b in range(len(x)):
                temp_x.append(self.decoder(x[id_b : id_b + 1]))
            x = torch.cat(temp_x, 0)

        return x
