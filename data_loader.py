import os
import numpy as np
import imageio
imageio.plugins.freeimage.download()
from collections import OrderedDict
import json

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def read_split(basedir, split, ignore_mask):
    cam_dict = json.load(open(os.path.join(basedir, split, 'cam_dict_norm.json')))

    imgsizes = []
    intrinsics = []
    poses = []
    imgfpaths = []
    maskfpaths = []
    for x in sorted(cam_dict.keys()):
        K = np.array(cam_dict[x]['K']).reshape((4, 4))
        W2C = np.array(cam_dict[x]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        W, H = cam_dict[x]['img_size']

        imgsizes.append(np.array([W, H]))
        intrinsics.append(K)
        poses.append(C2W)
        imgfpaths.append(os.path.join(basedir, split, 'image', x))

        mask_fpath = os.path.join(basedir, split, 'mask', x)
        if (not ignore_mask) and os.path.isfile(mask_fpath):
            maskfpaths.append(mask_fpath)
        else:
            maskfpaths.append(None)

    imgsizes = np.stack(imgsizes, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    poses = np.stack(poses, axis=0)

    return imgsizes, intrinsics, poses, imgfpaths, maskfpaths


def load_data(basedir, ignore_mask=True):
    imgsizes, intrinsics, poses, imgfpaths, maskfpaths = read_split(basedir, split='train', ignore_mask=ignore_mask)
    test_imgsizes, test_intrinsics, test_poses, test_imgfiles, test_maskfiles = read_split(basedir, split='test', ignore_mask=ignore_mask)
    val_imgsizes, val_intrinsics, val_poses, val_imgfiles, val_maskfiles = read_split(basedir, split='test', ignore_mask=ignore_mask)

    counts = [0] + [len(x) for x in [imgfpaths, val_imgfiles, test_imgfiles]]
    counts = np.cumsum(counts)
    i_split = [list(np.arange(counts[i], counts[i+1])) for i in range(3)]

    imgsizes = np.concatenate([imgsizes, val_imgsizes, test_imgsizes], 0)
    intrinsics = np.concatenate([intrinsics, val_intrinsics, test_intrinsics], 0)
    poses = np.concatenate([poses, val_poses, test_poses], 0)
    imgfpaths = imgfpaths + val_imgfiles + test_imgfiles
    maskfpaths = maskfpaths + val_maskfiles + test_maskfiles

    render_cams = None

    data = OrderedDict([('imgfpaths', imgfpaths),
                        ('maskfpaths', maskfpaths),
                        ('imgsizes', imgsizes),
                        ('intrinsics', intrinsics),
                        ('poses', poses),
                        ('i_train', i_split[0]),
                        ('i_val', i_split[1]),
                        ('i_test', i_split[2]),
                        ('render_cams', render_cams)])

    print('Data statistics:')
    print('\t # of training views: ', len(data['i_train']))
    print('\t # of validation views: ', len(data['i_val']))
    print('\t # of test views: ', len(data['i_test']))
    if data['render_cams'] is not None:
        print('\t # of render cameras: ', len(data['render_cams']))

    return data
