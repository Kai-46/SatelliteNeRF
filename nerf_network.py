import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import TINY_NUMBER


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** np.linspace(0., max_freq_log2, N_freqs, dtype=np.float32)
        else:
            self.freq_bands = np.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs, dtype=np.float32)
        self.freq_bands = self.freq_bands.tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        out = []
        if self.include_input:
            out.append(input)

        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        return out


class GeometryNet(nn.Module):
    def __init__(self, D=8, W=256, skips=[4],
                       input_ch=3, N_freqs=10,
                       output_ch=1, output_feature_ch=256):
        super().__init__()
        self.skips = skips

        self.input_ch = input_ch
        self.output_ch = output_ch
        self.output_feature_ch = output_feature_ch

        self.embedder = None
        if N_freqs > 0:
            self.embedder = Embedder(input_ch, max_freq_log2=N_freqs-1, N_freqs=N_freqs)
            input_ch = self.embedder.out_dim

        self.base_layers = []
        dim = input_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
            if i in self.skips and i != (D - 1):  # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)

        output_layers = [nn.Linear(dim, output_ch), ]
        self.output_layers = nn.Sequential(*output_layers)

        self.output_feature_layers = None
        if output_feature_ch > 0:
            output_feature_layers = [nn.Linear(dim, output_feature_ch), ]
            self.output_feature_layers = nn.Sequential(*output_feature_layers)

    def forward(self, input):
        '''
        :param input: [..., input_ch]
        :return [..., output_ch+output_feature_ch]
        '''
        if self.embedder is not None:
            input = self.embedder(input)

            # with torch.no_grad():
            #     input = self.embedder(input)
            # print(input.shape, type(input))

        base = self.base_layers[0](input)
        for i in range(len(self.base_layers) - 1):
            if i in self.skips:
                base = torch.cat((input, base), dim=-1)
            base = self.base_layers[i + 1](base)

        # sigma = torch.abs(self.output_layers(base))    # sigma must be positive
        sigma = F.relu(self.output_layers(base))    # sigma must be positive

        feature = None
        if self.output_feature_layers is not None:
            feature = self.output_feature_layers(base)
        return sigma.squeeze(-1), feature


class NerfNet(nn.Module):
    def __init__(self, geom_params={'D': 8, 'W': 256, 'skips': [4,],
                                    'input_ch': 3, 'N_freqs': 10,
                                    'output_ch': 1, 'output_feature_ch': 256},
                       radiance_params={'D': 3, 'W': 256,
                                        'viewdirs_N_freqs': 4}):
        super().__init__()
        self.geom_net = GeometryNet(**geom_params)

        self.embedder_viewdirs = Embedder(input_dim=3,
                                          max_freq_log2=radiance_params['viewdirs_N_freqs']-1,
                                          N_freqs=radiance_params['viewdirs_N_freqs'])

        radiance_layers = []
        dim = geom_params['output_feature_ch'] + self.embedder_viewdirs.out_dim
        for i in range(radiance_params['D']):
            if i == radiance_params['D'] - 1:
                out_dim = 3
                radiance_layers.append(
                    nn.Sequential(nn.Linear(dim, out_dim), nn.Sigmoid())
                )
            else:
                out_dim = radiance_params['W']
                radiance_layers.append(
                    nn.Sequential(nn.Linear(dim, out_dim), nn.ReLU())
                )
            dim = out_dim
        self.radiance_layers = nn.ModuleList(radiance_layers)
        print('Geometry layers: ', self.geom_net)
        print('Radiance layers: ', self.radiance_layers)

    def forward(self, pts, viewdirs, white_bkgd=False):
        '''
        :param pts:  [N_rays, N_samples, 3]
        :param viewdirs: [N_rays, N_samples, 3]
        :param white_bkgd: whether to use the white background trick
        :return:
        '''
        sigma, feature = self.geom_net(pts)
        assert (feature is not None)
        dists = torch.norm(pts[:, 1:, :] - pts[:, :-1, :], dim=-1)      # [N_rays, N_samples-1]
        # append an "infinite far" depth
        dists = torch.cat((dists, 1e10 * torch.ones_like(dists[:, 0:1])), dim=-1)  # [N_rays, N_samples]
        alpha = 1. - torch.exp(-sigma * dists)
        # Eq. (3): T
        T = torch.cumprod(1. - alpha + TINY_NUMBER, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
        T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        weights = alpha * T  # [N_rays, N_samples]

        tmp = torch.cat((feature, self.embedder_viewdirs(viewdirs)), dim=-1)
        for i in range(len(self.radiance_layers)):
            tmp = self.radiance_layers[i](tmp)
        rgb = torch.sum(weights.unsqueeze(-1) * tmp, dim=1)  # [N_rays, 3]

#        last_alpha = alpha[:, -1]
#        last_weight = weights[:, -1]
#        last_rgb = tmp[:, -1, :3]

        weights_sum = weights.sum(dim=-1, keepdim=True)
#        weights_norm = weights / weights_sum
#        surface_pts = torch.sum(weights_norm.unsqueeze(-1) * pts, dim=1)  # [N_rays, 3]

        if white_bkgd:
            rgb = rgb + (1. - weights_sum)
            # rgb = rgb + (1. - weights[:, :-1].sum(dim=-1, keepdim=True))

        # print('debug: ', weights_sum.min(), weights_sum.max(),
        #                  weights[:, :-1].sum(dim=-1, keepdim=True).min(), weights[:, :-1].sum(dim=-1, keepdim=True).max())
        # print('debug: ', rgb.shape, weights.shape, surface_pts.shape)
        # print('debug: ', rgb[0, :], tmp[0, :])
        ret = {
            'rgb': rgb,
            'weights': weights,
#            'surface_pts': surface_pts,
            'weights_sum': weights_sum,
#            'last_alpha': last_alpha,
#            'last_weight': last_weight,
#            'last_rgb': last_rgb
        }
        return ret


