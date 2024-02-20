from typing import List, Sequence, Optional, Union, Dict, Tuple

import os
import numpy as np
import torch
import torch.nn as nn

from plenoxels.models.density_fields import KPlaneDensityField
from plenoxels.models.kplane_field import KPlaneField
from plenoxels.models.semantic_field import FeatureField

from plenoxels.ops.activations import init_density_activation
from plenoxels.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from plenoxels.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from plenoxels.utils.timer import CudaTimer

import tinycudann as tcnn

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import plenoxels.models.vgg.VGGNet as VGGNet

from plenoxels.models.guided_filter.guided_filter import FastGuidedFilter2d
from kmeans_pytorch import kmeans
import models.dino.vision_transformer as vits

class LowrankModel(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)

        self.content_field = KPlaneField(
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )

        self.semantic_field = FeatureField(
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

        self.instanceNorm = torch.nn.InstanceNorm1d(self.content_field.geo_feat_dim, momentum=1e-3, track_running_stats=True)
        self.softmax = nn.Softmax(dim=1)

        self.render_only = kwargs['render_only']

        # construct style dictionary
        if self.render_only:
            # load image encoders
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.dinov1_model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
            self.dinov1_model.eval()
            self.dinov1_model.to(device)
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.dinov1_model.load_state_dict(state_dict, strict=True)

            decoder = VGGNet.decoder
            vgg = VGGNet.vgg
            vgg.load_state_dict(torch.load('plenoxels/models/vgg/checkpoints/vgg_normalised.pth'))
            vgg = nn.Sequential(*list(vgg.children())[:31])
            self.vggnet = VGGNet.Net(vgg, decoder).cuda()

            self.vgg_feat_dim = 128
            self.mean_conv = nn.Conv2d(self.vgg_feat_dim, self.vgg_feat_dim, kernel_size=3, stride = 2, padding = 1, bias=False, groups=self.vgg_feat_dim, device='cuda')
            self.mean_conv.weight = nn.Parameter(torch.ones_like(self.mean_conv.weight)/9, requires_grad=False)

            vit_transform = transforms.Compose([
                transforms.Resize([448,448]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            vgg_transform = transforms.Compose([
                transforms.Resize([448,448]),
                transforms.ToTensor()
            ])

            radius=30
            eps=1e-3

            GF = FastGuidedFilter2d(radius, eps, 2)

            style_path = kwargs['style_path']
            num_clusters = kwargs['num_clusters']
            local_global_blending_ratio = kwargs['local_global_blending_ratio']
            self.temperature = kwargs['temperature']

            style_imgs = os.listdir(style_path)

            # encode images & clustering
            with torch.no_grad():
                local_vgg_mean_1d_list = []
                local_vgg_std_1d_list = []
                style_dino_feature_1d_list = []
                vgg_feature_1d_list = []

                for style_img_path in style_imgs:
                    style_img_path = os.path.join(style_path, style_img_path)
                    style_img = Image.open(style_img_path).convert('RGB')
                    vit_style_img = vit_transform(style_img).unsqueeze(0).cuda()

                    dino_out = self.get_feature_map(self.dinov1_model, vit_style_img.squeeze())
                    fliped_dino_out = self.get_feature_map(self.dinov1_model, self.flip_image(vit_style_img).squeeze())
                    style_feature_map = dino_out
                    fliped_style_feature_map = fliped_dino_out
                    h, w = dino_out.shape[1], dino_out.shape[2]
                    fliped_style_feature_map_2d = fliped_style_feature_map.reshape(1, h, w, style_feature_map.shape[-1]).permute(0,3,1,2)
                    style_feature_map += self.flip_image(fliped_style_feature_map_2d).permute(0,2,3,1).reshape(style_feature_map.shape)
                    style_feature_map /= 2

                    tch_img = vit_style_img
                    tch_img = F.interpolate(tch_img, size=(113, 113), mode='nearest')
                    tch_mask = style_feature_map.reshape(1, 113, 113, 384).permute(0,3,1,2)

                    out = GF(tch_mask, tch_img).permute(0,2,3,1)
                    style_dino_feature_1d = out.reshape(-1,384)

                    style_vgg_feature = self.vggnet.encode_to_128_dim(vgg_transform(style_img).unsqueeze(0).cuda())

                    tch_img = vgg_transform(style_img).cuda().unsqueeze(0)
                    tch_mask = style_vgg_feature

                    out = GF(tch_mask, tch_img)
                    a = style_vgg_feature.reshape(128,-1)
                    b = out.reshape(128,-1)
                    print(torch.mean(torch.std(a, -1))/torch.mean(torch.std(b, -1)))
                    out = F.interpolate(out, size=(113,113), mode='nearest')
                    style_vgg_feature = out
                    style_vgg_feature_1d = out.reshape(128,-1).permute(1,0)

                    ids, center = kmeans(style_dino_feature_1d, num_clusters = num_clusters, distance = 'cosine', device = 'cuda')

                    local_vgg_mean_cluster = torch.zeros(num_clusters, self.vgg_feat_dim).cuda()
                    local_vgg_std_cluster = torch.zeros(num_clusters, self.vgg_feat_dim).cuda()

                    for i in range(num_clusters):
                        local_vgg_mean_cluster[i] = torch.mean(style_vgg_feature_1d[ids == i], 0)
                        local_vgg_std_cluster[i] = torch.std(style_vgg_feature_1d[ids == i], 0)

                    global_vgg_mean = torch.mean(style_vgg_feature_1d, 0).unsqueeze(0)
                    global_vgg_std = torch.std(style_vgg_feature_1d, 0).unsqueeze(0)

                    local_vgg_mean_cluster = local_vgg_mean_cluster * (1-local_global_blending_ratio) + global_vgg_mean * local_global_blending_ratio
                    local_vgg_std_cluster = local_vgg_std_cluster * (1-local_global_blending_ratio) + global_vgg_std * local_global_blending_ratio
                    
                    local_vgg_mean_1d_list.append(local_vgg_mean_cluster)
                    local_vgg_std_1d_list.append(local_vgg_std_cluster)
                    style_dino_feature_1d_list.append(center)
                    vgg_feature_1d_list.append(style_vgg_feature_1d)

                
                self.local_vgg_mean_1d = torch.cat(local_vgg_mean_1d_list, 0)
                self.local_vgg_std_1d = torch.cat(local_vgg_std_1d_list, 0)
                self.style_dino_feature_1d = torch.cat(style_dino_feature_1d_list, 0).cuda()
                self.vgg_feature_1d = torch.cat(vgg_feature_1d_list, 0)


    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb
    
    @staticmethod
    def render_feature(features: torch.Tensor, weights: torch.Tensor):
        comp_features = torch.sum(weights * features, dim=-2)
        return comp_features

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation
    
    @staticmethod
    def flip_image(img):
        assert(img.dim()==4)
        with torch.cuda.device_of(img):
            idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
        return img.index_select(3, idx)

    @staticmethod
    def get_feature_map(model, img, patch_size = 8):
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)

        features = model.get_features(img.to(torch.device("cuda")))
        features = features.reshape(1, 113, 113, features.shape[-1])

        return features

    def forward(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)

        field_out = self.content_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        rgb, density, grid_features, encoded_direction = field_out["rgb"], field_out["density"], field_out["features"], field_out["encoded_direction"]

        feature_out = self.semantic_field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        semantic_features = feature_out["semantic_features"]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        grid_features = grid_features.reshape(rgb.shape[0], rgb.shape[1], -1)
        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)

        rendered_color_features = self.render_feature(features=grid_features, weights=weights)
        rendered_semantic_features = self.render_feature(features=semantic_features, weights=weights.detach())

        self.instanceNorm(rendered_color_features.reshape(-1,128).T)

        # semantic matching & local AdaIN
        if self.render_only:
            with torch.no_grad():            
                corr_mat = torch.matmul(semantic_features.reshape(-1,384), self.style_dino_feature_1d.T)               

                corr_mat_soft = self.softmax(corr_mat/self.temperature)

                weighted_vgg_mean = torch.matmul(corr_mat_soft, self.local_vgg_mean_1d)
                weighted_vgg_std = torch.matmul(corr_mat_soft, self.local_vgg_std_1d)
                weighted_vgg_mean = torch.nan_to_num(weighted_vgg_mean)
                weighted_vgg_std = torch.nan_to_num(weighted_vgg_std)

                color_features_origin = grid_features

                blended_vgg_mean = weighted_vgg_mean 
                blended_vgg_std = weighted_vgg_std

                color_features_origin_norm = (self.instanceNorm(color_features_origin.reshape(-1, 128).T).T).reshape(color_features_origin.shape) 
                adain_feature = (color_features_origin_norm * blended_vgg_std.reshape(color_features_origin_norm.shape)) + blended_vgg_mean.reshape(color_features_origin_norm.shape)

                adain_feature_1d = adain_feature.view(-1, self.vgg_feat_dim)
                rgb = self.content_field.color_net(adain_feature_1d).float().view(adain_feature.shape[0], -1, 3)
                rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)


        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "color_feature_points": grid_features,
            "color_feature": rendered_color_features,
            "semantic_feature" : rendered_semantic_features
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_params = self.content_field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        
        semantic_model_params = self.semantic_field.get_params()
        semantic_field_params = semantic_model_params["field"]
        semantic_nn_params = semantic_model_params["nn"]
        semantic_other_params = semantic_model_params["other"]

        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},

            {"params": semantic_field_params, "lr": lr},
            {"params": semantic_nn_params, "lr": lr},
            {"params": semantic_other_params, "lr": lr},
        ]

