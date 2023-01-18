# Python
import os
import sys
import time
# import visdom
import random
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# NumPy and PyTorch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Cho et al. dependency
from support.WCMC import *

# Xu et al. dependency
from support.AdvMCD import *

# our strategy
from support.networks import InterpolationNet
from support.datasets import MSDenoiseDataset, DenoiseDataset
from support.utils import BasicArgumentParser
from support.interfaces import EnsembleKPCNInterface

# import logging
from tensorboardX import SummaryWriter

BS_VAL = 4

PRE_MODEL_FN = {
                    'dncnn_G': 'weights_full_3/KPCN_full_3.pth',
                    'dncnn_P': 'weights_full_3/KPCN_manif_p12_nogbuf_full_3.pth',
                    'backbone_diffuse': 'weights_full_3/KPCN_manif_p12_nogbuf_full_3.pth',
                    'backbone_specular': 'weights_full_3/KPCN_manif_p12_nogbuf_full_3.pth',
                }

PRE_MODEL_HALF_FN = {
                    'dncnn_G': 'weights_full_4/e6_KPCN_G_half.pth',
                    'dncnn_P': 'weights_full_4/e6_KPCN_P_half.pth',
                    'backbone_diffuse': 'weights_full_4/e6_KPCN_P_half.pth',
                    'backbone_specular': 'weights_full_4/e6_KPCN_P_half.pth',
                }

PRE_MODEL_FN_2 = {
                    'dncnn_G': 'weights_full_4/e7_KPCN_G.pth',
                    'dncnn_P': 'weights_full_4/e8_KPCN_P.pth',
                    'backbone_diffuse': 'weights_full_4/e8_KPCN_P.pth',
                    'backbone_specular': 'weights_full_4/e8_KPCN_P.pth',
                }

def logging(writer, epoch, s, split, relL2, best_relL2):
    writer.add_scalar('valid relL2 loss', relL2, epoch + s/split)
    writer.add_scalar('valid best relL2 loss', best_relL2, epoch + s/split)

def logging_training(writer, epoch, s, split, summaries):
    for m_losses in summaries:
        for key in m_losses:
            writer.add_scalar(key + ' train loss', m_losses[key], epoch + s/split)

def train_epoch_kpcn(epoch, interfaces, dataloaders, params, args):
    assert 'train' in dataloaders, "argument `dataloaders` dictionary should contain `'train'` key."
    assert 'data_device' in params, "argument `params` dictionary should contain `'data_device'` key."
    print('[][] Epoch %d' % (epoch))

    for itf in interfaces:
        itf.to_train_mode()
    

    for batch in tqdm(dataloaders['train'], leave=False, ncols=70):
        # Transfer data from the cpu to gpu memory
        for k in batch:
            if not batch[k].__class__ == torch.Tensor:
                continue
            batch[k] = batch[k].cuda(params['data_device'])

        # Main
        for itf in interfaces:
            itf.preprocess(batch)
            itf.train_batch(batch)
    
    summaries = []
    if not args.visual:
        for itf in interfaces:
            summaries.append(itf.get_epoch_summary(mode='train', norm=len(dataloaders['train'])))

    itf.epoch += 1
    itf.cnt = 0
    return summaries


def validate_kpcn(epoch, interfaces, dataloaders, params, split, args):
    assert 'val' in dataloaders, "argument `dataloaders` dictionary should contain `'train'` key."
    assert 'data_device' in params, "argument `params` dictionary should contain `'data_device'` key."
    print('[][] Validation (epoch %d split %d)' % (epoch, split))

    for itf in interfaces:
        itf.to_eval_mode()

    cnt = 0
    summaries = []
    with torch.no_grad():
        for batch in tqdm(dataloaders['val'], leave=False, ncols=70):
            # Transfer data from the cpu to gpu memory
            for k in batch:
                if not batch[k].__class__ == torch.Tensor:
                    continue
                batch[k] = batch[k].cuda(params['data_device'])

            # Main
            for itf in interfaces:
                itf.validate_batch(batch)

    for itr in interfaces:
        summaries.append(itf.get_epoch_summary(mode='eval', norm=len(dataloaders['val'])))

    return summaries


def train(interfaces, dataloaders, params, args):
    print('[] Experiment: `{}`'.format(args.desc))
    print('[] # of interfaces : %d'%(len(interfaces)))
    print('[] Model training start...')
    writer = SummaryWriter(args.summary + '/' + args.desc)
    
    # Start training
    m_val_err_diff, m_val_err_spec = 1e6, 1e6
    for epoch in range(args.start_epoch, args.num_epoch):
        if len(interfaces) == 1:
            save_fn = args.model_name + '.pth'
        else:
            raise NotImplementedError('Multiple interfaces are not supported yet')

        start_time = time.time()
        train_summaries = train_epoch_kpcn(epoch, interfaces, dataloaders, params, args)
        logging_training(writer, epoch, 0, 1, train_summaries)

        print('[][] Elapsed time: %d'%(time.time() - start_time))

        for i, itf in enumerate(interfaces):
            tmp_params = params.copy()
            tmp_params['vis'] = None

            state_dict = {
                'description': args.desc, #
                'start_epoch': epoch + 1,
                'model': str(itf),
                'params': tmp_params,
                'optims': itf.optims,
                'args': args,
                'best_err': itf.best_err
            }

            for model_name in itf.models:
                state_dict['state_dict_' + model_name] = itf.models[model_name].state_dict()

            if not args.not_save:
                torch.save(state_dict, os.path.join(args.save, 'e{}_'.format(str(epoch)) + save_fn))

        # Validate models
        if (epoch % args.val_epoch == args.val_epoch - 1):
            print('[][] Validation')
            summaries = validate_kpcn(epoch, interfaces, dataloaders, params, 0, args)

            for i, itf in enumerate(interfaces):
                if args.interpolate: summary = summaries[i]['m_val']
                else: summary = summaries[i]['m_val_err_diff'] + summaries[i]['m_val_err_spec']
                if summary < itf.best_err:
                    if args.interpolate:
                        itf.best_err = summary
                    else:
                        itf.best_err = summary
                        m_val_err_diff = summaries[i]['m_val_err_diff']
                        m_val_err_spec = summaries[i]['m_val_err_spec']


                    tmp_params = params.copy()
                    tmp_params['vis'] = None

                    state_dict = {
                        'description': args.desc, #
                        'start_epoch': epoch + 1,
                        'model': str(itf),
                        'params': tmp_params,
                        'optims': itf.optims,
                        'args': args,
                        'best_err': itf.best_err
                    }

                    for model_name in itf.models:
                        state_dict['state_dict_' + model_name] = itf.models[model_name].state_dict()

                    if not args.not_save:
                        torch.save(state_dict, os.path.join(args.save, save_fn))
                        print('[][] Model %s saved at epoch %d.'%(save_fn, epoch))
                if args.interpolate:
                    print('[][] Model {} RelMSE: {:.3f}e-3 \t Best RelMSE: {:.3f}e-3'.format(save_fn, summary*1000, itf.best_err*1000))
                else:
                    print('[][] Model {} err_diff L1: {:.3f}e-3 \t Best L1: {:.3f}e-3'.format(save_fn, summaries[i]['m_val_err_diff']*1000, m_val_err_diff*1000))
                    print('[][] Model {} err_spec L1: {:.3f}e-3 \t Best L1: {:.3f}e-3'.format(save_fn, summaries[i]['m_val_err_spec']*1000, m_val_err_spec*1000))

                logging(writer, epoch, 0, 1, summary, itf.best_err)
                

        # # Update schedulers
        for key in params:
            if 'sched_' in key:
                params[key].step()
    print('[] Training complete!')


def init_data(args):
    # Initialize datasets
    datasets = {}
    if 'full' in args.desc:
        print('load full dataset')
        datasets['train'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
            use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
        datasets['val'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
            use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
    else:
        print('load 8spp dataset')
        datasets['train'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
             use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
        datasets['val'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
             use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
    
    # Initialize dataloaders
    dataloaders = {}
    num_workers = 1
    dataloaders['train'] = DataLoader(
        datasets['train'], 
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=BS_VAL,
        num_workers=num_workers,
        pin_memory=True
    )
    return datasets, dataloaders


def init_model(dataset, args, rank=0):
    print('rank in init_model', rank)
    interfaces = []

    lr_pnets, lr_inets = args.lr_pnet, args.lr_inet
    pnet_out_sizes = args.pnet_out_size
    w_manifs = args.w_manif

    tmp = [lr_pnets, lr_inets, pnet_out_sizes, w_manifs]
    for lr_pnet, lr_inet, pnet_out_size, w_manif in list(itertools.product(*tmp)):
        # Initialize models (NOTE: modified for each model) 
        models = {}
        if args.train_branches:
            print('Train diffuse and specular branches indenpendently.')
        else:
            print('Train both branches with final radiance and single ensemble stream.')


        # set width of KPCN
        if 'half' in args.model_name:
            width = 50
            print('Use half of original capaticity, width :', width)
        else:
            width = 100
            print('Use all of original capaticity, width :', width)

        g_in = dataset['train'].dncnn_in_size
        if args.use_llpm_buf:
            if args.disentangle in ['m10r01', 'm11r01']:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size // 2
            else:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size
            # if args.manif_learn:
            n_in = dataset['train'].pnet_in_size
            n_out = pnet_out_size
            models['backbone_diffuse'] = PathNet(ic=n_in, outc=n_out)
            models['backbone_specular'] = PathNet(ic=n_in, outc=n_out)
            p_in = 10 + n_out + 1 + 1 #23
            # two stream denoisers
            if 'KPCN' in args.model_name:
                models['dncnn_G'] = KPCN(34, width=width)
                p_in = 23 # image 10 + pbuffer 12 + pvar 1
                models['dncnn_P'] = KPCN(p_in, width=width)
            elif 'ADV' in args.model_name:
                feat_ch = 7
                models['generator_diffuse_G'] = Generator(feat_ch=feat_ch)
                models['generator_specular_G'] = Generator(feat_ch=feat_ch)
                feat_ch = pnet_out_size + 1
                models['generator_diffuse_P'] = Generator(feat_ch=feat_ch)
                models['generator_specular_P'] = Generator(feat_ch=feat_ch)
                models['discriminator_diffuse'] = Discriminator()
                models['discriminator_specular'] = Discriminator()

            if args.feature:
                if 'KPCN' in args.model_name:
                    n_in = 34 + p_in - 20 + 6
                elif 'ADV' in args.model_name:
                    n_in = 7 + pnet_out_size + 1 + 6
            else:
                n_in = 6
            models['interpolate_diffuse'] = InterpolationNet(n_in, model_type=args.model_type)
            models['interpolate_specular'] = InterpolationNet(n_in, model_type=args.model_type)
            
        else:
            assert('should use llpm')
        
        # Load pretrained weights
        if len(list(itertools.product(*tmp))) == 1:
            model_fn = os.path.join(args.save, args.model_name + '.pth')
        else:
            model_fn = os.path.join(args.save, '%s_lp%f_pos%d_wgt%f.pth'%(args.model_name, lr_pnet, pnet_out_size, w_manif))
        assert args.start_epoch != 0 or not os.path.isfile(model_fn), 'Model %s already exists.'%(model_fn)
        is_pretrained = ((args.start_epoch != 0) and os.path.isfile(model_fn)) or args.load
        # print('is pretrained', args.start_epoch != 0, os.path.isfile(model_fn), args.load)
        if is_pretrained:
            # loading pretrained weight
            if args.load:
                # path for loading pretrained weight
                for model_name in models:
                    # if model_name in PRE_MODEL_FN:
                    if model_name in PRE_MODEL_HALF_FN:
                    # if model_name in PRE_MODEL_FN_2:
                        print(model_name)
                        # ck_m = torch.load(PRE_MODEL_FN[model_name], map_location='cuda:{}'.format(args.device_id))
                        ck_m = torch.load(PRE_MODEL_HALF_FN[model_name], map_location='cuda:{}'.format(args.device_id))
                        # ck_m = torch.load(PRE_MODEL_FN_2[model_name], map_location='cuda:{}'.format(args.device_id))
                        if 'dncnn' in model_name: pre_model_name = 'dncnn'
                        else: pre_model_name = model_name
                        try:
                            models[model_name].load_state_dict(ck_m['state_dict_' + pre_model_name])
                        except RuntimeError:
                            new_state_dict = OrderedDict()
                            for k, v in ck_m['state_dict_' + pre_model_name].items():
                                # name = k[7:]
                                name = k
                                new_state_dict[name] = v
                            models[model_name].load_state_dict(new_state_dict)
                print('Pretraining weights are loaded initially')
                

            else:
            # finetuning
                ck = torch.load(model_fn, map_location='cuda:{}'.format(args.device_id))
                for model_name in models:
                    try:
                        models[model_name].load_state_dict(ck['state_dict_' + model_name])
                    except RuntimeError:
                        new_state_dict = OrderedDict()
                        for k, v in ck['state_dict_' + model_name].items():
                            # name = k[7:]
                            name = k
                            new_state_dict[name] = v
                        models[model_name].load_state_dict(new_state_dict)
                print('Pretraining weights are loaded for resume')
        else:
            print('Train models from scratch.')

        # Use GPU parallelism if needed
        if args.single_gpu:
            print('Data Sequential')
            for model_name in models:
                # models[model_name] = models[model_name].cuda(args.device_id)
                models[model_name] = models[model_name].cuda()
        else:
            # multi_gpu support with DDP
            print('Data Parallel')
            if args.distributed:
                if args.world_size == 1:
                    print('Single CUDA machine detected')
                    for model_name in models:
                        models[model_name] = models[model_name].cuda(args.device_id)
                elif args.world_size > 1:
                    print('Data parallel & Distributed')
                    print('%d CUDA machines detected' % (torch.cuda.device_count()))
                    for model_name in models:
                        models[model_name].cuda()
                        models[model_name] == DDP(models[model_name], device_ids=[rank], output_device=rank, find_unused_parameters=False)
            elif torch.cuda.device_count() == 1:
                print('Single CUDA machine detected')
                for model_name in models:
                    models[model_name] = models[model_name].cuda()
            elif torch.cuda.device_count() > 1:
                print('%d CUDA machines detected' % (torch.cuda.device_count()))
                for model_name in models:
                    models[model_name] = nn.DataParallel(models[model_name], output_device=1).cuda()
            else:
                assert False, 'No detected GPU device.'
        
        # Initialize optimizers
        optims = {}
        for model_name in models:
            if 'dncnn' in model_name:
                lr = args.lr_dncnn
            elif 'backbone' in model_name:
                lr = lr_pnet
            elif 'interpolat' in model_name:
                lr  = lr_inet
            optims['optim_' + model_name] = optim.Adam(models[model_name].parameters(), lr=lr)
            print('optim for', model_name, lr)
            if not is_pretrained or 'error' in model_name or 'interpolate' in model_name:
            # if not is_pretrained:
                # print('continue!')
                continue
            
            if not args.load:
                if 'optims' in ck:
                    state = ck['optims']['optim_' + model_name].state_dict()
                elif 'optim_' + model_name in ck['params']:
                    state = ck['params']['optim_' + model_name].state_dict()
                else:
                    print('No state for the optimizer for %s, use the initial optimizer and learning rate.'%(model_name))
                    continue
            else:
                # if model_name in PRE_MODEL_FN:
                if 'half' in args.model_name:
                    MODEL_FN = PRE_MODEL_HALF_FN
                else:
                    MODEL_FN = PRE_MODEL_FN_2
                if model_name in MODEL_FN:
                    # print(model_name)
                    # ck_m = torch.load(PRE_MODEL_FN[model_name], map_location='cuda:{}'.format(args.device_id))
                    ck_m = torch.load(MODEL_FN[model_name], map_location='cuda:{}'.format(args.device_id))
                    # ck_m = torch.load(PRE_MODEL_FN_2[model_name], map_location='cuda:{}'.format(args.device_id))
                    if 'dncnn' in model_name: pre_model_name = 'dncnn'
                    else: pre_model_name = model_name
                    if 'optims' in ck_m:
                        state = ck_m['optims']['optim_' + pre_model_name].state_dict()
                    elif 'optim_' + model_name in ck_m['params']:
                        state = ck_m['params']['optim_' + pre_model_name].state_dict()
                    else:
                        print('No state for the optimizer for %s, use the initial optimizer and learning rate.'%(model_name))
                        continue

            print('use ckpt', args.lr_ckpt)
            if not args.lr_ckpt:
                print('Set the new learning rate %.3e for %s.'%(lr, model_name))
                state['param_groups'][0]['lr'] = lr
            else:
                print('Use the checkpoint (%s) learning rate for %s.'%(model_fn, model_name))
            print('load', model_name)
            optims['optim_' + model_name].load_state_dict(state)

            # to remove error https://github.com/pytorch/pytorch/issues/80809
            optims['optim_' + model_name].param_groups[0]['capturable'] = True

        # Initialize losses (NOTE: modified for each model)
        loss_funcs = {
            'l_diffuse': nn.L1Loss(),
            'l_specular': nn.L1Loss(),
            'l_recon': nn.L1Loss(),
            'l_test': RelativeMSE()
        }
        if 'ADV' in args.model_name:
            loss_funcs['l_gan'] = WGANLoss()
            loss_funcs['l_gp'] = GradientPenaltyLoss()

        if args.manif_learn:
            if args.manif_loss == 'FMSE':
                loss_funcs['l_manif'] = FeatureMSE(non_local = not args.local)
                print('Manifold loss: FeatureMSE')
            elif args.manif_loss == 'GRS':
                loss_funcs['l_manif'] = GlobalRelativeSimilarityLoss()
                print('Manifold loss: Global Relative Similarity') 
        else:
            print('Manifold loss: None (i.e., ablation study)')

        # multi_gpu for loss functions
        for loss_name in loss_funcs:
            loss_funcs[loss_name].cuda()

        # Initialize a training interface (NOTE: modified for each model)
        if 'KPCN' in args.model_name:
            itf = EnsembleKPCNInterface(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn)
        elif 'ADV' in args.model_name:
            pass
        if is_pretrained and not args.load:
            # TODO: needs change in automatically updating best_err
            print('Use the checkpoint best error %.3e'%(args.best_err))
            itf.best_err = args.best_err 
        interfaces.append(itf)
    
    # Initialize a visdom visualizer object
    params = {
        'plots': {},
        'data_device': 1 if torch.cuda.device_count() > 1 and not args.single_gpu else args.device_id,
    }
    
    # Make the save directory if needed
    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    return interfaces, params


def main(args):
    # Set random seeds
    random.seed("Inyoung Cho, Yuchi Huo, Sungeui Yoon @ KAIST")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True  #torch.backends.cudnn.deterministic = True
    # Get ready

    dataset, dataloaders = init_data(args)
    interfaces, params = init_model(dataset, args)
    train(interfaces, dataloaders, params, args)

if __name__ == "__main__":

    BS_VAL = 4 # validation set batch size

    parser = BasicArgumentParser()
    parser.add_argument('--desc', type=str, required=True, 
                        help='short description of the current experiment.')
    parser.add_argument('--lr_dncnn', type=float, default=1e-4, 
                        help='learning rate of PathNet.')
    parser.add_argument('--lr_pnet', type=float, nargs='+', default=[0.0001], 
                        help='learning rate of PathNet.')
    parser.add_argument('--lr_ckpt', action='store_true',
                        help='')
    parser.add_argument('--best_err', type=float, required=False)
    parser.add_argument('--pnet_out_size', type=int, nargs='+', default=[3], 
                        help='# of channels of outputs of PathNet.')
    parser.add_argument('--manif_loss', type=str, required=False,
                        help='`FMSE` or `GRS`')
    
    parser.add_argument('--train_branches', action='store_true',
                        help='train the diffuse and specular branches independently.')
    parser.add_argument('--use_llpm_buf', action='store_true',
                        help='use the llpm-specific buffer.')
    parser.add_argument('--manif_learn', action='store_true',
                        help='use the manifold learning loss.')
    parser.add_argument('--w_manif', type=float, nargs='+', default=[0.1], 
                        help='ratio of the manifold learning loss to \
                        the reconstruction loss.')
    
    parser.add_argument('--disentangle', type=str, default='m11r11',
                        help='`m11r11`, `m10r01`, `m10r11`, or `m11r01`')

    parser.add_argument('--single_gpu', action='store_true',
                        help='use only one GPU.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id')

    parser.add_argument('--not_save', action='store_true',
                        help='do not save checkpoint (debugging purpose).')
    parser.add_argument('--local', action='store_true', help='device id')

    # new arguments
    parser.add_argument('--lr_inet', type=float, nargs='+', default=[0.0001], 
                        help='learning rate of InterpolationNet.')

    parser.add_argument('--model_type', type=str, default='conv5',
                        help='model type for interpolation weight')
    parser.add_argument('--interpolate', action='store_true',
                        help='train to reconstruct interpolated result')
    parser.add_argument('--load', action='store_true',
                        help='load the pretrained denoisers for each stream')
    parser.add_argument('--fix', action='store_true',
                        help='fix the denoisers when training')
    parser.add_argument('--feature', action='store_true', 
                        help='feed features to InterpolationNet')
    parser.add_argument('--weight', type=int, default=0,
                        help='weight of full training denoisers')
    
    
    args = parser.parse_args()
    
    if args.manif_learn and not args.use_llpm_buf:
        raise RuntimeError('The manifold learning module requires a llpm-specific buffer.')
    if args.manif_learn and not args.manif_loss:
        raise RuntimeError('The manifold learning module requires a manifold loss.')
    if not args.manif_learn and args.manif_loss:
        raise RuntimeError('A manifold loss is not necessary when the manifold learning module is opted out.')
    if args.manif_learn and args.manif_loss not in ['GRS', 'FMSE']:
        raise RuntimeError('Argument `manif_loss` should be either `FMSE` or `GRS`')
    if args.disentangle not in ['m11r11', 'm10r01', 'm10r11', 'm11r01']:
        raise RuntimeError('Argument `disentangle` should be either `m11r11`, `m10r01`, `m10r11`, or `m11r01`')
    for s in args.pnet_out_size:
        if args.disentangle != 'm11r11' and s % 2 != 0:
            raise RuntimeError('Argument `pnet_out_size` should be a list of even numbers')
    if args.single_gpu:
        print('device:', args.device_id)
        torch.cuda.set_device(f'cuda:{args.device_id}')
    else:
        args.world_size = torch.cuda.device_count()
    main(args)
