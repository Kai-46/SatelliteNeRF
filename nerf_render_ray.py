import torch
from collections import OrderedDict
from utils import TINY_NUMBER

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################
def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER      # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00       # prevent outlier samples
    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
    denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def intersect_sphere(rays_o, rays_d):
    '''
    rays_o, rays_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    assume camera is outside unit sphere
    '''
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d
    ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)

    mask = p_norm_sq < 1.

    p_norm_sq = torch.clamp(p_norm_sq, 0., 1.)      # consider the case where the ray does not intersect the sphere
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos
    return d1 - d2, d1 + d2, mask


def render_rays(ray_batch, models,
                N_samples,
                N_importance=0,
                det=False,
                white_bkgd=False):
    '''
    :param ray_batch: {'rays_o': [N_rays, 3] , 'rays_d': [N_rays, 3], 'view_dir': [N_rays, 3]}
    :param models:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''
    near_depth, far_depth, sphere_intersect_mask = intersect_sphere(rays_o=ray_batch['rays_o'], rays_d=ray_batch['rays_d'])
    depth_range = far_depth - near_depth

    step = (far_depth - near_depth) / (N_samples-1)
    z_vals = torch.stack([near_depth+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]

    rays_d = ray_batch['rays_d'].unsqueeze(1).repeat(1, N_samples, 1) # [N_rays, N_samples, 3]
    rays_o = ray_batch['rays_o'].unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * rays_d + rays_o       # [N_rays, N_samples, 3]
    viewdirs = -rays_d / rays_d.norm(dim=-1, keepdim=True)    # ---> camera

    outputs_coarse = models['net_coarse'](pts, viewdirs, white_bkgd=white_bkgd)
#    outputs_coarse['depth'] = ((outputs_coarse['surface_pts'] - ray_batch['rays_o']) / ray_batch['rays_d']).mean(dim=-1)
    outputs_coarse['depth'] = (outputs_coarse['weights'] * z_vals).sum(dim=-1)

    ret = OrderedDict([('outputs_coarse', outputs_coarse), ])

    if N_importance > 0:
        # detach since we would like to decouple the coarse and fine networks
        weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]
        # take mid-points of depth samples
        z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
        z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                               N_samples=N_importance, det=det)  # [N_rays, N_importance]
        z_vals = torch.cat((z_vals, z_samples), dim=-1)         # [N_rays, N_samples + N_importance]
        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)

        rays_d = ray_batch['rays_d'].unsqueeze(1).repeat(1, N_samples+N_importance, 1)
        rays_o = ray_batch['rays_o'].unsqueeze(1).repeat(1, N_samples+N_importance, 1)
        pts = z_vals.unsqueeze(2) * rays_d + rays_o            # [N_rays, N_samples + N_importance, 3]
        viewdirs = -rays_d / rays_d.norm(dim=-1, keepdim=True)  # ---> camera

        outputs_fine = models['net_fine'](pts, viewdirs, white_bkgd=white_bkgd)
        # outputs_fine['depth'] = ((outputs_fine['surface_pts'] - ray_batch['rays_o']) / ray_batch['rays_d']).mean(dim=-1)
        outputs_fine['depth'] = (outputs_fine['weights'] * z_vals).sum(dim=-1)
        ret['outputs_fine'] = outputs_fine

        # save memory
        outputs_fine.pop('weights')
    else:
        ret['outputs_fine'] = None

    # save memory
    outputs_coarse.pop('weights')

    ### extra outputs for debugging purposes
    # ret['outputs_fine']['sphere_intersect_mask'] = sphere_intersect_mask
    # ret['outputs_fine']['depth_range'] = depth_range
    return ret
