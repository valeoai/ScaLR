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


import torch
import torch.nn as nn
import torchvision.transforms as T
import models.dinov2_vision_transformer as dinov2_vit


DINOv2_MODELS = {
    "dinov2_vit_small_p14": ("dinov2_vits14", 14, 384),
    "dinov2_vit_base_p14": ("dinov2_vitb14", 14, 768),
    "dinov2_vit_large_p14": ("dinov2_vitl14", 14, 1024),
}


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

        # ViT parameters
        model_name, patch_size, embed_dim = DINOv2_MODELS.get(
            config["image_backbone"]["images_encoder"]
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.which_feature = config["image_backbone"]["feat"]
        print("Image teacher:")
        print(f"==> model_name: {model_name}")
        print(f"==> patch_size: {patch_size}")
        print(f"==> embed_dim: {embed_dim}")
        assert config["point_backbone"]["nb_class"] == embed_dim

        # Compute feature size
        height, width = config["image_backbone"]["im_size"]
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        self.f_height = height // self.patch_size
        self.f_width = width // self.patch_size

        # Load ViT
        self.encoder = dinov2_vit.__dict__[model_name](
            patch_size=patch_size,
            pretrained=True,
        )

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

            output = self.encoder.forward_get_last_n(x)
            feat = output[self.which_feature]
            x = torch.cat(feat, dim=2)

            # Remove the CLS token and reshape the patch token features.
            x = (
                x[:, 1:, :]
                .transpose(1, 2)
                .view(batch_size, self.embed_dim, self.f_height, self.f_width)
            )

            # Go through decoder
            x = self.decoder(x)

        return x
