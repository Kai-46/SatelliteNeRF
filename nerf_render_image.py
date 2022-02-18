import torch
import numpy as np
from collections import OrderedDict
import imageio
import os
from nerf_sample_ray import RaySamplerSingleImage
from nerf_render_ray import render_rays
from utils import to8b, colorize_np
import time


def render_single_image(ray_sampler, models, chunk_size,
                        N_samples,
                        N_importance=0,
                        white_bkgd=False):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param models:  {'net_coarse':  , 'net_fine': }
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''
    ray_batch = ray_sampler.get_all()

    all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                           ('outputs_fine', OrderedDict())])

    N_rays = ray_sampler.H * ray_sampler.W
    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
                if torch.cuda.is_available():
                    chunk[k] = chunk[k].cuda()
            else:
                chunk[k] = None

        with torch.no_grad():
            ret = render_rays(chunk, models, N_samples,
                              N_importance=N_importance,
                              det=True,
                              white_bkgd=white_bkgd)
            # key_to_extract = ['rgb', 'depth', 'weights_sum']
            # for k in sorted(ret['outputs_coarse'].keys()):
            #     if k not in key_to_extract:
            #         ret['outputs_coarse'].pop(k)
            # if ret['outputs_fine'] is not None:
            #     for k in sorted(ret['outputs_fine'].keys()):
            #         if k not in key_to_extract:
            #             ret['outputs_fine'].pop(k)
        if i == 0:
            for k in ret['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())         # cache chunk results on cpu

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    # use mask
    if ray_batch['mask'] is not None:
        mask = ray_batch['mask'].cpu().reshape((ray_sampler.H, ray_sampler.W, 1))
    else:
        mask = None

    # merge chunk results and reshape
    for k in all_ret['outputs_coarse']:
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape(
                                        (ray_sampler.H, ray_sampler.W, -1))
        if mask is not None:
            tmp = mask * tmp + (1. - mask) * torch.zeros_like(tmp)
        all_ret['outputs_coarse'][k] = tmp.squeeze()

    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape(
                                        (ray_sampler.H, ray_sampler.W, -1))

            if mask is not None:
                tmp = mask * tmp + (1. - mask) * torch.zeros_like(tmp)
            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret


def maskout_pixels(heatmap, mask=None):
    if mask is None:
        return heatmap
    else:
        if len(heatmap.shape) == 3 and len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        return heatmap * mask + np.ones_like(heatmap) * (1 - mask)


def batch_render_images(ray_samplers, out_dir,
                        models, chunk_size,
                        N_samples,
                        N_importance=0,
                        white_bkgd=False):
    '''
    :param render_cams:
    :param out_dir:
    :param gt_imgs:
    :param gt_img_paths:
    :return: no return; simply write results to out_dir
    '''
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for i in range(len(ray_samplers)):
        print('Rendering image {}/{}'.format(i, len(ray_samplers)))
        t0 = time.time()

        ray_sampler = ray_samplers[i]
        ret = render_single_image(ray_sampler=ray_sampler, models=models,
                                  chunk_size=chunk_size,
                                  N_samples=N_samples,
                                  N_importance=N_importance,
                                  white_bkgd=white_bkgd)

        dt = time.time() - t0
        print('\t Spent {} seconds'.format(dt))

        which_outputs = 'outputs_fine'
        if ret['outputs_fine'] is None:
            which_outputs = 'outputs_coarse'

        ret = ret[which_outputs]
        rgb = ret['rgb'].numpy()
        depth = ret['depth'].numpy()

        fname = '{:06}.png'.format(i)
        gt_img, gt_mask = ray_sampler.get_img_and_mask()
        if gt_img is not None:
            fname = os.path.basename(ray_sampler.img_fpath)
            if fname.endswith('.exr'):
                imageio.imwrite(os.path.join(out_dir, 'gt_' + fname), gt_img)
            else:
                imageio.imwrite(os.path.join(out_dir, 'gt_' + fname), to8b(gt_img))

        if gt_mask is not None:
            rgb = maskout_pixels(rgb, gt_mask)
            imageio.imwrite(os.path.join(out_dir, 'mask_' + fname[:-4]+'.png'), to8b(gt_mask))

        if fname.endswith('.exr'):
            imageio.imwrite(os.path.join(out_dir, 'nerf_' + fname), rgb)
        else:
            imageio.imwrite(os.path.join(out_dir, 'nerf_' + fname), to8b(rgb))

        depth_vis = colorize_np(depth, cmap_name='jet', mask=gt_mask, append_cbar=True)
        imageio.imwrite(os.path.join(out_dir, 'depth_' + fname), to8b(depth_vis))

        # sphere_intersect_mask = ret['sphere_intersect_mask'].float().numpy()
        # imageio.imwrite(os.path.join(out_dir, 'sph_int_mask_' + fname), to8b(sphere_intersect_mask))

        # depth_range = ret['depth_range'].numpy()
        # depth_range_vis = colorize_np(depth_range, cmap_name='jet', mask=gt_mask, append_cbar=True)
        # imageio.imwrite(os.path.join(out_dir, 'depth_range_' + fname), to8b(depth_range_vis))

        frames.append(to8b(rgb))

    imageio.mimwrite(os.path.join(out_dir, 'video.mp4'), frames, fps=3, quality=8)
    print('Done rendering', out_dir)


if __name__ == '__main__':
    from data_loader import load_data
    from nerf_model import create_nerf, load_nerf
    from run_nerf import config_parser

    parser = config_parser()
    args = parser.parse_args()

    data = load_data(args.datadir)
    models = create_nerf(args)

    start = -1
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]
    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        print('Reloading from', fpath)
        models = load_nerf(models, fpath)
        start = int(fpath[-10:-4])

    split = 'test'
    out_dir = os.path.join(args.basedir, args.expname, '{}set_{:06d}'.format(split, start))
    print('Rendering {} set...'.format(split))
    ray_samplers = []
    for i in data['i_{}'.format(split)]:
        ray_samplers.append(RaySamplerSingleImage(img_size=data['imgsizes'][i],
                                                  K=data['intrinsics'][i],
                                                  C2W=data['poses'][i],
                                                  img_fpath=data['imgfpaths'][i],
                                                  mask_fpath=data['maskfpaths'][i], downsample_factor=4))
    batch_render_images(ray_samplers, out_dir=out_dir,
                        models=models, chunk_size=args.chunk_size,
                        N_samples=args.N_samples,
                        N_importance=args.N_importance,
                        white_bkgd=args.white_bkgd)

