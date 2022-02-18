import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

import time
import numpy as np
from collections import OrderedDict

import torch
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from tensorboardX import SummaryWriter

from data_loader import load_data
from nerf_render_ray import render_rays
from nerf_render_image import render_single_image
from nerf_model import create_nerf, save_nerf, load_nerf
from nerf_sample_ray import RaySamplerSingleImage
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, gray2rgb


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')

    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--downsample_factor", type=int, default=1, help='image downsampling factor')

    # model size
    parser.add_argument("--netdepth_coarse", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth_coarse", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')

    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')

    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--N_iters_perturb", type=int, default=1000,
                        help='perturb and center-crop at first 1000 iterations to prevent training from getting stuck')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='apply the trick to avoid fitting to white background')

    # no training; render only
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_train", action='store_true', help='render the training set')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')

    # no training; extract mesh only
    parser.add_argument("--mesh_only", action='store_true',
                        help='do not optimize, extract mesh from pretrained model')
    parser.add_argument("--N_pts", type=int, default=256,
                        help='voxel resolution; N_pts * N_pts * N_pts')
    parser.add_argument("--mesh_thres", type=str, default='10,20,30,40,50',
                        help='threshold(s) for mesh extraction; can use multiple thresholds')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    print(parser.format_values())

    ### Create log dir and copy the config file
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.basedir, args.expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    ### Load data
    data = load_data(args.datadir)

    ### Create nerf
    models = create_nerf(args)

    ### Load pretrained model
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

    ### Start training
    # create ray_samplers for training images
    ray_samplers = []
    for idx in data['i_train']:
        ray_samplers.append(RaySamplerSingleImage(img_size=data['imgsizes'][idx],
                                                  K=data['intrinsics'][idx],
                                                  C2W=data['poses'][idx],
                                                  img_fpath=data['imgfpaths'][idx],
                                                  mask_fpath=data['maskfpaths'][idx],
                                                  downsample_factor=args.downsample_factor))

    writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))
    for global_step in range(start+1, start+1+args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        models['optimizer'].zero_grad()

        # Sample random ray batch
        if global_step <= args.N_iters_perturb:         # use cropped data at beginning
            center_crop = True
        else:
            center_crop = False

        i = np.random.randint(low=0, high=len(ray_samplers))
        ray_batch = ray_samplers[i].random_sample(args.N_rand,
                                                  center_crop=center_crop)
        # print('args.N_rand: ', args.N_rand, ' , ray_batch size: ', ray_batch['rays_d'].shape[0])
        if torch.cuda.is_available():
            for k in ray_batch:
                if torch.is_tensor(ray_batch[k]):
                    ray_batch[k] = ray_batch[k].cuda()

        ret = render_rays(ray_batch=ray_batch,
                          models=models,
                          N_samples=args.N_samples,
                          N_importance=args.N_importance,
                          det=args.det,
                          white_bkgd=args.white_bkgd)

        # compute loss
        loss = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb'], ray_batch['mask'])
        loss.backward()

        scalars_to_log['coarse/loss'] = loss.item()
        scalars_to_log['coarse/pnsr'] = mse2psnr(loss.item())

        if ret['outputs_fine'] is not None:
            loss = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb'], ray_batch['mask'])
            loss.backward()

            scalars_to_log['fine/loss'] = loss.item()
            scalars_to_log['fine/pnsr'] = mse2psnr(loss.item())

        models['optimizer'].step()
        models['scheduler'].step()

        scalars_to_log['lr'] = models['scheduler'].get_lr()[0]
        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        # Rest is logging
        if global_step % args.i_print == 0 or global_step < 10:
            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            print(logstr)

        if global_step % args.i_weights == 0 and global_step > 0:
            print('Saving checkpoints...')
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            save_nerf(models, fpath)

        if global_step % args.i_img == 0:
            '''
            print('Logging a random validation view...')
            idx = np.random.choice(data['i_val'])
            tmp_ray_sampler = RaySamplerSingleImage(img_size=data['imgsizes'][idx],
                                                    K=data['intrinsics'][idx],
                                                    C2W=data['poses'][idx],
                                                    img_fpath=data['imgfpaths'][idx],
                                                    mask_fpath=data['maskfpaths'][idx],
                                                    half_res=False)
            log_view_to_tb(writer, global_step, args, models, tmp_ray_sampler, prefix='val/')
            '''

            print('Logging a random training view...')
            idx = np.random.choice(data['i_train'])
            tmp_ray_sampler = RaySamplerSingleImage(img_size=data['imgsizes'][idx],
                                                    K=data['intrinsics'][idx],
                                                    C2W=data['poses'][idx],
                                                    img_fpath=data['imgfpaths'][idx],
                                                    mask_fpath=data['maskfpaths'][idx],
                                                    downsample_factor=args.downsample_factor)
            print('tmp_ray_sampler image size: ', tmp_ray_sampler.H, tmp_ray_sampler.W)
            log_view_to_tb(writer, global_step, args, models, tmp_ray_sampler, prefix='train/')


def log_view_to_tb(writer, global_step, args, models, ray_sampler, prefix=''):
    ret = render_single_image(ray_sampler=ray_sampler,
                              models=models, chunk_size=args.chunk_size,
                              N_samples=args.N_samples,
                              N_importance=args.N_importance,
                              white_bkgd=args.white_bkgd)
    gt_img, gt_mask = ray_sampler.get_img_and_mask()
    assert ((gt_img is not None) and (gt_mask is None))
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    rgb_im = torch.cat((rgb_im, img_HWC2CHW(ret['outputs_coarse']['rgb'])), dim=-1)

    depth_im = img_HWC2CHW(colorize(ret['outputs_coarse']['depth'], cmap_name='jet', append_cbar=True,
                                    mask=gt_mask))
    writer.add_image(prefix + 'coarse/depth', depth_im, global_step)

    # print('debug: ', ret['outputs_coarse'].keys())
    weights_sum_im = img_HWC2CHW(gray2rgb(ret['outputs_coarse']['weights_sum']))
    writer.add_image(prefix + 'coarse/weights_sum', weights_sum_im, global_step)

    # last_alpha_im = img_HWC2CHW(gray2rgb(ret['outputs_coarse']['last_alpha']))
    # writer.add_image(prefix + 'coarse/last_alpha', last_alpha_im, global_step)

    # last_weight_im = img_HWC2CHW(gray2rgb(ret['outputs_coarse']['last_weight']))
    # writer.add_image(prefix + 'coarse/last_weight', last_weight_im, global_step)

    # last_rgb_im = img_HWC2CHW(ret['outputs_coarse']['last_rgb'])
    # writer.add_image(prefix + 'coarse/last_rgb', last_rgb_im, global_step)

    if ret['outputs_fine'] is not None:
        rgb_im = torch.cat((rgb_im, img_HWC2CHW(ret['outputs_fine']['rgb'])), dim=-1)

        depth_im = img_HWC2CHW(colorize(ret['outputs_fine']['depth'], cmap_name='jet', append_cbar=True,
                                        mask=gt_mask))
        writer.add_image(prefix + 'fine/depth', depth_im, global_step)

        weights_sum_im = img_HWC2CHW(gray2rgb(ret['outputs_fine']['weights_sum']))
        writer.add_image(prefix + 'fine/weights_sum', weights_sum_im, global_step)

        # last_alpha_im = img_HWC2CHW(gray2rgb(ret['outputs_fine']['last_alpha']))
        # writer.add_image(prefix + 'fine/last_alpha', last_alpha_im, global_step)

        # last_weight_im = img_HWC2CHW(gray2rgb(ret['outputs_fine']['last_weight']))
        # writer.add_image(prefix + 'fine/last_weight', last_weight_im, global_step)

        # last_rgb_im = img_HWC2CHW(ret['outputs_fine']['last_rgb'])
        # writer.add_image(prefix + 'fine/last_rgb', last_rgb_im, global_step)

    # add comparison of rgb images
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)


if __name__ == '__main__':
    train()
