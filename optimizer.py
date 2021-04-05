"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if args.sgd:
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.adam:
        amsgrad = False
        if args.amsgrad:
            amsgrad = True
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=amsgrad
                               )
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
             math.pow(1 - epoch / args.max_epoch,
                      args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in loaded_dict:
        if k not in net_state_dict or net_state_dict[k].size() != loaded_dict[k].size():
            logging.info("Uable to load parameter %s", k)
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            logging.info("Skipped loading parameter %s", k)
        '''elif 'bn2.semantic' in k and 'num_batches_tracked' not in k:
            new_loaded_dict[k] = loaded_dict[k.replace('.semantic', '')]
        elif 'bn2.traversability' in k and 'num_batches_tracked' not in k:
            new_loaded_dict[k] = loaded_dict[k.replace('.traversability', '')]
        elif 'bn3.semantic' in k and 'num_batches_tracked' not in k:
            new_loaded_dict[k] = loaded_dict[k.replace('.semantic', '')]
        elif 'bn3.traversability' in k and 'num_batches_tracked' not in k:
            new_loaded_dict[k] = loaded_dict[k.replace('.traversability', '')]
        elif 'aspp2' in k:
            new_loaded_dict[k] = loaded_dict[k.replace('aspp2', 'aspp')]
        elif 'bot_fine2' in k:
            new_loaded_dict[k] = loaded_dict[k.replace('bot_fine2', 'bot_fine')]
        elif 'final2' in k and k != 'module.final2.6.weight':
            new_loaded_dict[k] = loaded_dict[k.replace('final2', 'final')]'''
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
