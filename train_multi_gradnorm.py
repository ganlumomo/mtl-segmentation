"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
from apex import amp

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval_multi, fast_hist
import datasets
import loss
import network
import optimizer

from torchvision import transforms
from PIL import Image

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')

parser.add_argument('--cv', type=int, default=None,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
args = parser.parse_args()
args.best_record1 = {'epoch': -1, 'iter': 0, 'val_loss1': 1e10, 'acc1': 0,
                    'acc_cls1': 0, 'mean_iu1': 0, 'fwavacc1': 0}
args.best_record2 = {'epoch': -1, 'iter': 0, 'val_loss2': 1e10, 'acc2': 0,
                    'acc_cls2': 0, 'mean_iu2': 0, 'fwavacc2': 0}
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ and args.apex:
    args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

if args.apex:
    # Check that we are running with cuda as distributed is only supported for cuda.
    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)
    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

def main():
    """
    Main Function
    """

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)

    tasks = ['semantic', 'traversability']
    criterion, criterion2, criterion_val = loss.get_loss(args, tasks=tasks)
    net = network.get_net(args, criterion=criterion, criterion2=criterion2, tasks=tasks)
    optim, scheduler = optimizer.get_optimizer(args, net)

    if args.fp16:
        net, optim = amp.initialize(net, optim, opt_level="O1")

    net = network.wrap_network_in_dataparallel(net, args.apex)
    if args.snapshot:
        optimizer.load_weights(net, optim,
                               args.snapshot, args.restore_optimizer)

    torch.cuda.empty_cache()
    # Main Loop
    initial_task_loss = []
    for epoch in range(args.start_epoch, args.max_epoch):
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)

        scheduler.step()
        initial_task_loss = train(train_loader, net, optim, epoch, writer, tasks, initial_task_loss)
        if args.apex:
            train_loader.sampler.set_epoch(epoch + 1)
        validate(val_loader, net, criterion_val,
                 optim, epoch, writer)
        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                if args.apex:
                    train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()


def train(train_loader, net, optim, curr_epoch, writer, tasks, initial_task_loss):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    GradNormLoss = torch.nn.L1Loss()

    train_main_loss1 = AverageMeter()
    train_main_loss2 = AverageMeter()
    train_main_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        inputs, gts, _img_name, inputs2, gts2, _img_name2 = data

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs, gts = inputs.cuda(), gts.cuda()
        inputs2, gts2 = inputs2.cuda(), gts2.cuda()
        
        # DEBUG
        '''img = transforms.ToPILImage()(inputs[0,:].squeeze_(0))
        img.save('images/inputs.png')
        img = transforms.ToPILImage()(gts[0,:].type(torch.DoubleTensor))
        img.save('images/gts.png')
        img = transforms.ToPILImage()(inputs2[0,:].squeeze_(0))
        img.save('images/inputs2.png')
        img = transforms.ToPILImage()(gts2[0,:].type(torch.DoubleTensor))
        img.save('images/gts2.png')'''

        optim.zero_grad()

        main_loss1 = net(inputs, gts=gts, task='semantic')
        main_loss2 = net(inputs2, gts=gts2, task='traversability')
        #main_loss = main_loss1 + main_loss2

        task_loss = []
        task_loss.append(main_loss1)
        task_loss.append(main_loss2)
        task_loss = torch.stack(task_loss)
        weighted_task_loss = torch.mul(net.module.task_weights, task_loss)

	# Initialize the initial loss L(0) if t=0
        if curr_iter == 0:
            initial_task_loss = task_loss.data

	# get the total loss
        main_loss = torch.sum(weighted_task_loss)

        if args.apex:
            log_main_loss1 = main_loss1.clone().detach_()
            log_main_loss2 = main_loss2.clone().detach_()
            log_main_loss = main_loss.clone().detach_()
            torch.distributed.all_reduce(log_main_loss1, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(log_main_loss2, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(log_main_loss, torch.distributed.ReduceOp.SUM)
            log_main_loss1 = log_main_loss1 / args.world_size
            log_main_loss2 = log_main_loss2 / args.world_size
            log_main_loss = log_main_loss / args.world_size
        else:
            main_loss1 = main_loss1.mean()
            main_loss2 = main_loss2.mean()
            main_loss = main_loss.mean()
            log_main_loss1 = main_loss1.clone().detach_()
            log_main_loss2 = main_loss2.clone().detach_()
            log_main_loss = main_loss.clone().detach_()
        train_main_loss1.update(log_main_loss1.item(), batch_pixel_size)
        train_main_loss2.update(log_main_loss2.item(), batch_pixel_size)
        train_main_loss.update(log_main_loss.item(), batch_pixel_size)
        
        if args.fp16:
            with amp.scale_loss(main_loss, optim) as scaled_loss:
                scaled_loss.backward()
            '''with amp.scale_loss(main_loss1, optim) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
            with amp.scale_loss(main_loss2, optim) as scaled_loss:
                scaled_loss.backward()'''
        else:
            main_loss.backward(retain_graph=True)
            '''main_loss1.backward(retain_graph=True)
            main_loss2.backward()'''

	# Set the gradients of w_i(t) according to GradNorm loss
        net.module.task_weights.grad.data = net.module.task_weights.grad.data * 0.0
        if True:
            W = net.module.get_last_shared_layer()
            norms = []
            for t in range(len(task_loss)):
                gygw = torch.autograd.grad(task_loss[t], W.parameters(), retain_graph=True)
                norms.append(torch.norm(torch.mul(net.module.task_weights[t], gygw[0]), p=2))
            norms = torch.stack(norms)
            mean_norm = torch.mean(norms)
            #print('G_w(t): {}'.format(norms))
              
            # compute the inverse training rate r_i(t)
            loss_ratio = task_loss / initial_task_loss
            inverse_train_rate = loss_ratio / torch.mean(loss_ratio)

            # compute the GradNorm loss
            constant_term = mean_norm * (inverse_train_rate ** 0.15)
            constant_term = constant_term.detach()
            # this is the GradNorm loss itself
            grad_norm_loss = torch.add(GradNormLoss(norms[0], constant_term[0]), GradNormLoss(norms[1], constant_term[1]))
            # compute the gradient for the task weights
            net.module.task_weights.grad = torch.autograd.grad(grad_norm_loss, net.module.task_weights)[0]

        optim.step()

        # renormalize task weights
        normalize_coeff = 2.0 / torch.sum(net.module.task_weights.data, dim=0)
        net.module.task_weights.data = net.module.task_weights.data * normalize_coeff

        curr_iter += 1

        if args.local_rank == 0:
            msg = '[epoch {}], [iter {} / {}], [loss1 {:0.6f}], [loss2 {:0.6f}], [w1 {:0.6f}], [w2 {:0.6f}], [main loss {:0.6f}], [lr {:0.6f}]'.format(
                curr_epoch, i + 1, len(train_loader), train_main_loss1.avg, train_main_loss2.avg,
                net.module.task_weights.data[0], net.module.task_weights.data[1], train_main_loss.avg,
                optim.param_groups[-1]['lr'])

            logging.info(msg)

            # Log tensorboard metrics for each iteration of the training phase
            writer.add_scalar('training/weight1', net.module.task_weights.data[0], curr_iter)
            writer.add_scalar('training/weight2', net.module.task_weights.data[1], curr_iter)
            writer.add_scalar('training/loss1', (train_main_loss1.val), curr_iter)
            writer.add_scalar('training/loss2', (train_main_loss2.val), curr_iter)
            writer.add_scalar('training/loss', (train_main_loss.val), curr_iter)
            writer.add_scalar('training/lr', optim.param_groups[-1]['lr'], curr_iter)

        if i > 5 and args.test_mode:
            return
    return initial_task_loss


def validate(val_loader, net, criterion, optim, curr_epoch, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss1 = AverageMeter()
    val_loss2 = AverageMeter()
    iou_acc1 = 0
    iou_acc2 = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image, img_names, inputs2, gt_image2, img_names2 = data
        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()
        inputs2, gt_cuda2 = inputs2.cuda(), gt_image2.cuda()

        with torch.no_grad():
            output1, _ = net(inputs)  # output = (1, 19, 713, 713)
            _, output2 = net(inputs2)

        assert output1.size()[2:] == gt_image.size()[1:]
        assert output1.size()[1] == args.dataset_cls.num_classes1
        assert output2.size()[2:] == gt_image2.size()[1:]
        assert output2.size()[1] == args.dataset_cls.num_classes2

        val_loss1.update(criterion(output1, gt_cuda).item(), batch_pixel_size)
        val_loss2.update(criterion(output2, gt_cuda2).item(), batch_pixel_size)
        predictions1 = output1.data.max(1)[1].cpu()
        predictions2 = output2.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions1, img_names])
            dump_images.append([gt_image2, predictions2, img_names2])

        iou_acc1 += fast_hist(predictions1.numpy().flatten(), gt_image.numpy().flatten(),
                             args.dataset_cls.num_classes1)
        iou_acc2 += fast_hist(predictions2.numpy().flatten(), gt_image2.numpy().flatten(),
                             args.dataset_cls.num_classes2)
        del output1, output2, val_idx, data

    if args.apex:
        iou_acc_tensor1 = torch.cuda.FloatTensor(iou_acc1)
        iou_acc_tensor2 = torch.cuda.FloatTensor(iou_acc2)
        torch.distributed.all_reduce(iou_acc_tensor1, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(iou_acc_tensor2, op=torch.distributed.ReduceOp.SUM)
        iou_acc1 = iou_acc_tensor1.cpu().numpy()
        iou_acc2 = iou_acc_tensor2.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval_multi(args, net, optim, val_loss1, val_loss2, iou_acc1, iou_acc2, dump_images,
                      writer, curr_epoch, args.dataset_cls)

    return val_loss1.avg, val_loss2.avg


if __name__ == '__main__':
    main()
