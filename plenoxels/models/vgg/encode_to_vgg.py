import os
import torch
import shutil
import plenoxels.models.vgg.VGGNet as VGGNet
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms

from plenoxels.models.guided_filter.guided_filter import GuidedFilter2d, FastGuidedFilter2d
import torch.nn.functional as F

transform = transforms.ToTensor()

def generate_vgg_feature_maps(datadir, dsample):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = VGGNet.decoder
    vgg = VGGNet.vgg
    
    vgg.load_state_dict(torch.load('plenoxels/models/vgg/checkpoints/vgg_normalised.pth'))

    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = VGGNet.Net(vgg, decoder)
    network.eval()
    network.to(device)
    
    img_dir = os.path.join(datadir, 'images_{}'.format(dsample))
    img_list = os.listdir(img_dir)
    os.makedirs(os.path.join(datadir, 'vgg_features_{}'.format(dsample)), exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(img_list, desc="generating vgg feature maps"):
            data_img_path = os.path.join(img_dir, img_path)
            img_origin = Image.open(data_img_path)
            img = img_origin.convert('RGB')
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            vgg_feature = network.encode_to_128_dim(img).cpu()

            radius=30
            eps=1e-3
            GF = FastGuidedFilter2d(radius, eps, 2)

            tch_img = img
            tch_mask = vgg_feature.cuda()

            out = GF(tch_mask, tch_img)

            out = F.interpolate(out, (img_origin.size[1],img_origin.size[0]), mode='bilinear')

            torch.save(out.half().cpu(), os.path.join(datadir, 'vgg_features_{}/{}.pt'.format(dsample, img_path[:-4])))
