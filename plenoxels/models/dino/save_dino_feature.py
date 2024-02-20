# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from tqdm import tqdm



import torch
import torch.nn as nn

from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import plenoxels.models.dino.vision_transformer as vits
from plenoxels.models.guided_filter.guided_filter import GuidedFilter2d, FastGuidedFilter2d
import torch.nn.functional as F

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
upsample = nn.UpsamplingBilinear2d(scale_factor=8)


def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


def get_feature_map(model, img):
    patch_size = 8
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    # attentions = model.get_last_selfattention(img.to(device))
    features = model.get_features(img.cuda())
    # features = features.reshape(1, w_featmap, h_featmap, features.shape[-1])
    features = features.reshape(1, 95, 127, features.shape[-1])

    return features


def generate_semantic_feature_maps(datadir, dsample):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dinov1_model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
    dinov1_model.eval()
    dinov1_model.to(device)
    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    dinov1_model.load_state_dict(state_dict, strict=True)

    transform = pth_transforms.Compose([
        pth_transforms.Resize([378,504]),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_dir = os.path.join(datadir, 'images_{}'.format(dsample))
    img_list = os.listdir(img_dir)
    os.makedirs(os.path.join(datadir, 'dino_features_{}'.format(dsample)), exist_ok=True)

    with torch.no_grad():

        for img_path in tqdm(img_list, desc="generating dino feature maps"):
            data_img_path = os.path.join(img_dir, img_path)
            img_origin = Image.open(data_img_path)
            img = img_origin.convert('RGB')
            img = transform(img).unsqueeze(0).cuda()

            feature_map = get_feature_map(dinov1_model, img.squeeze())
            fliped_feature_map = get_feature_map(dinov1_model, flip_image(img).squeeze())
            fliped_feature_map = fliped_feature_map.reshape(1, 95, 127, 384).permute(0,3,1,2)
            feature_map += flip_image(fliped_feature_map).permute(0,2,3,1).reshape(feature_map.shape)
            feature_map /= 2
            feature_map_upsample = F.interpolate(feature_map.reshape(1, 95, 127, 384).permute(0,3,1,2), (378, 504), mode='bilinear')

            radius=30
            eps=1e-3

            GF = FastGuidedFilter2d(radius, eps, s=2)

            tch_img = img
            tch_mask = feature_map_upsample
            out = GF(tch_mask, tch_img)

            out = F.interpolate(out, (img_origin.size[1],img_origin.size[0]), mode='bilinear')
            
            torch.save(out.half().cpu(), os.path.join(datadir, 'dino_features_{}/{}.pt'.format(dsample, img_path[:-4])))

