import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
import os


def load_img(img_fpath, mask_fpath=None):
    im = imageio.imread(img_fpath).astype(np.float32)
    # ldr or hdr
    if img_fpath.endswith('.png') or img_fpath.endswith('.jpg'):
        im = im / 255.
    elif img_fpath.endswith('.exr'):
        print('Clipping hdr images to [0,1]')
        im = np.clip(im, 0., 1.)
    # rgba
    if im.shape[-1] == 4:
        alpha = im[:, :, 3:4]
        im = im[:, :, :3] * alpha + np.ones_like(im[:, :, :3]) * (1. - alpha)
    # read mask and maskout background as white
    if (mask_fpath is not None) and (os.path.isfile(mask_fpath)):
        print('Loading mask and masking images with white background!')
        mask = imageio.imread(mask_fpath).astype(np.float32) / 255.
        im = im * mask[:, :, np.newaxis] + np.ones_like(im) * (1. - mask[:, :, np.newaxis])
    return im, None


def get_rays(u, v, K, C2W):
    '''
    :param u: 1D array; [N, ]
    :param v: 1D array; [N, ]
    :param K: 4x4 matrix
    :param C2W: 4x4 matrix
    :return:
    '''
    u = u.astype(dtype=np.float64) + 0.5                # add half pixel
    v = v.astype(dtype=np.float64) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, N)

    rays_d = np.matmul(np.linalg.inv(K[:3, :3]), pixels)
    rays_d = np.matmul(C2W[:3, :3], rays_d)  # (3, N)
    rays_d = rays_d.transpose((1, 0))  # (N, 3)

    rays_o = C2W[:3, 3].reshape((1, 3))
    dist = np.linalg.norm(rays_o.flatten())
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (N, 3)

    # print('dist: ', dist)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    ## shift rays towards camera
    if dist > 1e3:
        rays_o = rays_o + rays_d * (dist - 5.)

    return rays_o.astype(np.float32), rays_d.astype(np.float32)


class RaySamplerSingleImage(object):
    def __init__(self, img_size, K, C2W, img_fpath=None, mask_fpath=None, half_res=False, downsample_factor=1):
        super().__init__()
        self.W, self.H = img_size
        self.K = np.copy(K)
        self.C2W = np.copy(C2W)

        self.img_fpath = img_fpath
        self.mask_fpath = mask_fpath

        self.img = None
        self.mask = None
        if self.img_fpath is not None:
            self.img, self.mask = load_img(self.img_fpath, self.mask_fpath)

        if half_res:
            downsample_factor = 2

        # half-resolution output
        if downsample_factor != 1:
            self.W = self.W // downsample_factor
            self.H = self.H // downsample_factor
            self.K[:2, :3] /= downsample_factor
            if self.img is not None:
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            if self.mask is not None:
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_AREA)

        if self.img is not None:
            self.img = self.img.reshape((-1, 3))
        if self.mask is not None:
            self.mask = self.mask.reshape((-1))

    def get_img_and_mask(self):
        img = self.img
        mask = self.mask
        if img is not None:
            img = img.reshape((self.H, self.W, 3))
        if mask is not None:
            mask = mask.reshape((self.H, self.W))
        return img, mask

    def get_all(self):
        select_inds = np.arange(self.H * self.W)
        v = select_inds // self.W
        u = select_inds - v * self.W
        rays_o, rays_d = get_rays(u, v, self.K, self.C2W)
        ret = OrderedDict([
            ('rays_o', rays_o),
            ('rays_d', rays_d),
            ('rgb', self.img),
            ('mask', self.mask)
        ])
        # convert to torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] * self.W + u[select_inds]     # convert back to original image
        else:
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        v = select_inds // self.W
        u = select_inds - v * self.W

        rays_o, rays_d = get_rays(u, v, self.K, self.C2W)
        # print('here: ', rays_o.shape, rays_d.shape, u.shape, v.shape)

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        ret = OrderedDict([
            ('rays_o', rays_o),
            ('rays_d', rays_d),
            ('rgb', rgb),
            ('mask', mask)
        ])

        # convert to torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])

        return ret
