import torch
from collections import OrderedDict
from nerf_network import IDRNet, NerfNet


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


def create_nerf(args):
    '''
    :param args.
    :return:
    '''
    models = OrderedDict([('net_coarse', None),
                          ('net_fine', None),
                          ('optimizer', None),
                          ('scheduler', None)])

    # coarse net
    models['net_coarse'] = NerfNet()

    # fine net
    if args.N_importance > 0:
        models['net_fine'] = NerfNet()

    # move to gpu if possible
    if torch.cuda.is_available():
        models['net_coarse'] = models['net_coarse'].cuda()
        if models['net_fine'] is not None:
            models['net_fine'] = models['net_fine'].cuda()

    # optimizer and learning rate scheduler
    learnable_params = list(models['net_coarse'].parameters())
    if models['net_fine'] is not None:
        learnable_params += list(models['net_fine'].parameters())

    optimizer = torch.optim.Adam(learnable_params, lr=args.lrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrate_decay_steps,
                                                gamma=args.lrate_decay_factor)
    models['optimizer'] = optimizer
    models['scheduler'] = scheduler

    return models


def save_nerf(models, filename):
    to_save = OrderedDict()

    name_list = ['optimizer', 'scheduler', 'net_coarse',]
    if models['net_fine'] is not None:
        name_list.append('net_fine')

    for name in name_list:
        to_save[name] = models[name].state_dict()
    torch.save(to_save, filename)


def load_nerf(models, filename):
    to_load = torch.load(filename)

    name_list = ['optimizer', 'scheduler', 'net_coarse',]
    if models['net_fine'] is not None:
        name_list.append('net_fine')

    for name in name_list:
        models[name].load_state_dict(to_load[name])

    return models
