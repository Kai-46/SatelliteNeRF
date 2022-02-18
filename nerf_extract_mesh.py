import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import os

import mcubes

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################################################################################################################
#
########################################################################################################################

# helper function for test-time evaluation of RGBA for a bunch of scene points
def batch_eval_pts(pts_batch, models,
                   chunk_size, run_net_fine=True):
    '''
    :param pts_batch: [N_pts, 3]
    :param models:  {'net_coarse':  , 'net_fine': }
    :param chunk_size:
    :return:
    '''
    viewdirs = torch.zeros_like(pts_batch)
    pts = embed_input(pts_batch, viewdirs, models)

    all_ret = []
    N_rays = pts.shape[0]
    for i in range(0, N_rays, chunk_size):
        chunk = pts[i:i+chunk_size]
        if torch.cuda.is_available():
            chunk = chunk.cuda()

        with torch.no_grad():
            if run_net_fine:
                ret = models['net_fine'](chunk)    # [N, 4]
            else:
                ret = models['net_coarse'](chunk)

        # cache chunk results on cpu
        all_ret.append(ret.cpu())

    all_ret = torch.cat(all_ret, dim=0)

    return all_ret


def get_field_function(x_pts, y_pts, z_pts, models, chunk_size, run_net_fine=True):
    Nx = x_pts.shape[0]
    Ny = y_pts.shape[0]
    Nz = z_pts.shape[0]

    # [Nx, Ny, Nz, 3]
    xyz = np.stack(np.meshgrid(x_pts, y_pts, z_pts, indexing='ij'), axis=-1).astype(np.float32)
    query_pts = torch.from_numpy(xyz.reshape((-1, 3)))      # [Nx*Ny*Nz, 3]

    ret = batch_eval_pts(pts_batch=query_pts,
                         models=models,
                         chunk_size=chunk_size,
                         run_net_fine=run_net_fine)  # [N*N*N, 4]

    field = ret.numpy().reshape((Nx, Ny, Nz, -1))    # [Nx, Ny, Nz, 4]

    return xyz, field


def extract_mesh(N, models, chunk_size, thresholds, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    axis_pts = np.linspace(-1.05, 1.05, N)
    query_pts = np.stack(np.meshgrid(axis_pts, axis_pts, axis_pts), axis=-1).astype(np.float32)
    query_pts = torch.from_numpy(query_pts.reshape((-1, 3)))      # [N*N*N, 3]

    run_net_fine = True
    if models['net_fine'] is None:
        run_net_fine = False

    ret = batch_eval_pts(pts_batch=query_pts,
                         models=models,
                         chunk_size=chunk_size,
                         run_net_fine=run_net_fine)  # [N*N*N, 4]

    ret = ret.numpy().reshape((N, N, N, -1))    # [N, N, N, 4]
    sigma = ret[..., -1]  # clipping; [N, N, N]

    # #
    # plt.figure()
    # plt.hist(sigma.flatten(), log=True)
    # # plt.hist(sigma.flatten())
    # plt.xlabel('sigma')
    # plt.ylabel('freq.')
    # plt.savefig(os.path.join(out_dir, 'sigma_distribution.png'))
    # plt.close()

    for thres in thresholds:
        print('fraction occupied', np.mean(sigma > thres))
        vertices, triangles = mcubes.marching_cubes(sigma, thres)
        print('vertices: ', vertices.shape, ' triangles: ', triangles.shape)

        # change to the original unit
        vertices = vertices.astype(dtype=np.int)
        # somehow we need to swtich x and y
        vertices = np.stack([axis_pts[vertices[:, i]] for i in [1, 0, 2]], axis=1)

        # save mesh
        mesh_fname = os.path.join(out_dir, 'mesh_N_{}_T_{}.obj'.format(N, thres))
        mcubes.export_obj(vertices, triangles, mesh_fname)



########################################################################################################################
# visualize sigma with plane sweeping
########################################################################################################################

from nerf_sample_ray import RaySamplerSingleImage
from utils import colorize_np, to8b
import imageio



def visualize_sigma(camera_params, models, num_planes, chunk_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ray_sampler = RaySamplerSingleImage(camera_params, half_res=True)
    ray_batch = ray_sampler.get_all()

    depth = ray_batch['depth']
    # near_depth = torch.where(depth-1.>0., depth-1., 0.1*depth)
    near_depth = depth - 1.2
    far_depth = depth + 1.2
    step = (far_depth - near_depth) / (num_planes-1)
    depth_vals = [near_depth + i*step for i in range(num_planes)]

    query_pts = []
    for z in depth_vals:
        pts = ray_batch['ray_o'] + ray_batch['ray_d'] * z.unsqueeze(1)
        query_pts.append(pts)

    query_pts = torch.stack(query_pts, dim=0)       # [D, H*W, 3]
    query_pts = query_pts.reshape((-1, 3))      # [D*H*W, 3]

    print('query_pts bounding box: ', torch.min(query_pts, dim=0)[0], torch.max(query_pts, dim=0)[0])

    run_net_fine = True
    if models['net_fine'] is None:
        run_net_fine = False

    ret = batch_eval_pts(pts_batch=query_pts,
                         models=models,
                         chunk_size=chunk_size,
                         run_net_fine=run_net_fine)  # [D*H*W, 4]
    sigma = ret[..., -1].numpy()    # [D*H*W,]

    # log-space visualize
    sigma = np.log10(sigma + 1e-5)
    sigma, cbar = colorize_np(sigma.reshape((ray_sampler.H, -1)), cmap_name='hot')

    # # plot volumes in 3d
    # xyz = query_pts.reshape((num_planes, ray_sampler.H, ray_sampler.W, -1))
    # color = sigma.reshape((num_planes, ray_sampler.H, ray_sampler.W, -1))
    # # downsample to avoid crashing slow plotting
    # xyz = xyz[:, ::2, ::2, :].reshape((-1, 3))
    # color = color[:, ::2, ::2, :].reshape((-1, 3))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(xs=xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2], c=color.reshape((-1, 3)))
    # ax.scatter(xs=xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2])
    # plt.savefig(os.path.join(out_dir, 'sigma_volumes.png'))

    #
    sigma = sigma.reshape((num_planes, ray_sampler.H, ray_sampler.W, -1))

    #
    frames = []
    for i in range(sigma.shape[0]):
        im = np.concatenate((sigma[i], np.zeros((ray_sampler.H, 5, 3), dtype=np.float32), cbar), axis=1)
        im = to8b(im)

        imageio.imwrite(os.path.join(out_dir, 'sigma_{}.png'.format(i)), im)
        frames.append(im)

    imageio.mimwrite(os.path.join(out_dir, 'video.mp4'), frames, fps=6, quality=8)


if __name__ == '__main__':
    from data_loader import load_data
    from nerf_model import create_nerf, load_nerf
    from run_nerf import config_parser

    parser = config_parser()
    args = parser.parse_args()

    data = load_data(args.datadir, args.scene, testskip=1)
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

    ###
    # out_dir = os.path.join(args.basedir, args.expname, 'sigma_{:06d}'.format(start))
    # visualize_sigma(camera_params=data['cameras'][0],
    #                 models=models, num_planes=100,
    #                 chunk_size=args.chunk_size, out_dir=out_dir)

    print('Meshing only')
    thresholds = [float(x) for x in args.mesh_thres.split(',')]
    out_dir = os.path.join(args.basedir, args.expname, 'meshonly_{:06d}'.format(start))
    extract_mesh(N=args.N_pts,
                 models=models,
                 chunk_size=args.chunk_size,
                 thresholds=thresholds,
                 out_dir=out_dir)

    ## Extract field function
    # print('Extracting field function only')
    # out_dir = os.path.join(args.basedir, args.expname, 'field_{:06d}'.format(start))
    # os.makedirs(out_dir, exist_ok=True)
    #
    # from ply_np_converter import np2ply
    #
    # N = 128
    # for axis_range in [(-1., 1.), (-1.5, 1.5), (-2., 2.)]:
    #     print('axis_range: ', axis_range)
    #
    #     axis_pts = np.linspace(axis_range[0], axis_range[1], N)
    #     xyz, field = get_field_function(x_pts=axis_pts, y_pts=axis_pts, z_pts=axis_pts,
    #                                     models=models, chunk_size=args.chunk_size, run_net_fine=True)
    #     np.save(os.path.join(out_dir, 'B_{:3.2f}_xyz_field.npy'.format(axis_range[1])),
    #             np.concatenate((xyz, field), axis=-1))
    #
    #     xyz = xyz.reshape((-1, 3))
    #     # take log-sigma
    #     log_sigma = np.log10(field[..., -1] + 1e-5)     # [Nx, Ny, Nz]
    #     H = 512
    #     log_sigma, cbar = colorize_np(log_sigma.reshape((H, -1)), cmap_name='hot')
    #     log_sigma = log_sigma.reshape((-1, 3))
    #     np.save(os.path.join(out_dir, 'B_{:3.2f}_log-sigma.npy'.format(axis_range[1])),
    #             np.concatenate((xyz, log_sigma), axis=-1))
    #     imageio.imwrite(os.path.join(out_dir, 'B_{:3.2f}_log-sigma_colorbar.png'.format(axis_range[1])),
    #                     to8b(cbar))
    #
    #     np2ply(vertex=xyz, color=np.uint8(log_sigma * 255.),
    #            out_ply=os.path.join(out_dir, 'B_{:3.2f}_log-sigma.ply'.format(axis_range[1])))
    #
    #     rgb = field[..., :3].reshape((-1, 3))
    #     np.save(os.path.join(out_dir, 'B_{:3.2f}_rgb.npy'.format(axis_range[1])),
    #             np.concatenate((xyz, rgb), axis=-1))
    #
    #     np2ply(vertex=xyz, color=np.uint8(rgb * 255.),
    #            out_ply=os.path.join(out_dir, 'B_{:3.2f}_rgb.ply'.format(axis_range[1])))
