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
import configs
from support.networks import PathNet
# our strategy
from support.networks import AdvKPCN, NewAdvKPCN_2, PixelDiscriminator, NewAdvKPCN_1
from support.datasets import MSDenoiseDataset, DenoiseDataset
from support.utils import BasicArgumentParser
from support.losses import RelativeMSE, FeatureMSE, GlobalRelativeSimilarityLoss
from support.interfaces import AdvKPCNInterface, NewAdvKPCNInterface, NewAdvKPCNInterface1, NewAdvKPCNInterface2
from train_kpcn import validate_kpcn, train, train_epoch_kpcn

# Gharbi et al. dependency
sys.path.insert(1, configs.PATH_SBMC)
# try:
#     from sbmc import KPCN
# except ImportError as error:
#     print('Put appropriate paths in the configs.py file.')
#     raise

# for multi_gpu and amp support from KISTI
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
# import logging
from tensorboardX import SummaryWriter
from train_kpcn import logging, logging_training

BS_VAL = 4
# def setup_logger():
#     # create logger
#     logger = logging.getLogger(__package__)
#     # logger.setLevel(logging.DEBUG)
#     logger.setLevel(logging.INFO)

#     # create console handler and set level to debug
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)

#     # create formatter
#     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

#     # add formatter to ch
#     ch.setFormatter(formatter)

#     # add ch to logger
#     logger.addHandler(ch)


def init_data(args):
    # Initialize datasets
    datasets = {}
    if 'full' in args.desc:
        print('load full dataset')
        datasets['train'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
            use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
        datasets['val'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
            use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
    else:
        print('load 8spp dataset')
        datasets['train'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
             use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
        datasets['val'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
             use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)

    if args.distributed:
        train_sampler = DistributedSampler(datasets['train'], shuffle=False)
        val_sampler = DistributedSampler(datasets['val'], shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Initialize dataloaders
    dataloaders = {}
    if args.distributed: num_workers = 1# torch.cuda.device_count()
    else: num_workers = 1
    dataloaders['train'] = DataLoader(
        datasets['train'], 
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=BS_VAL,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler
    )
    return datasets, dataloaders


def init_model(dataset, args, rank=0):
    print('rank in init_model', rank)
    interfaces = []

    lr_pnets = args.lr_pnet
    pnet_out_sizes = args.pnet_out_size
    w_manifs = args.w_manif

    tmp = [lr_pnets, pnet_out_sizes, w_manifs]
    for lr_pnet, pnet_out_size, w_manif in list(itertools.product(*tmp)):
        # Initialize models (NOTE: modified for each model) 
        models = {}
        if args.train_branches:
            print('Train diffuse and specular branches indenpendently.')
        else:
            print('Post-train two branches of KPCN.')
            
        if args.use_llpm_buf:
            if args.disentangle in ['m10r01', 'm11r01']:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size // 2
            else:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size
            print('adv', n_in, pnet_out_size)
            if not args.use_single:
                print('adv type', args.type)
                # print(args.type == 'new_adv_1')
                if args.type == 'new_adv_1':
                    if args.manif_learn:
                        n_in = dataset['train'].pnet_in_size
                        n_out = pnet_out_size
                        models['backbone_diffuse'] = PathNet(ic=n_in, outc=n_out)
                        models['backbone_specular'] = PathNet(ic=n_in, outc=n_out)
                        p_in = 10 + n_out + 1 #23
                    else:
                        p_in = 46
                    models['dncnn'] = NewAdvKPCN_1(35, p_in, gen_activation=args.activation, disc_activtion=args.disc_activation, output_type=args.output_type, strided_down=args.strided_down, interpolation=args.interpolation, revise=args.revise, weight=args.weight)
                elif args.type == 'new_adv_2':
                    if args.manif_learn:
                        n_in = dataset['train'].pnet_in_size
                        n_out = pnet_out_size
                        models['backbone_diffuse'] = PathNet(ic=n_in, outc=n_out)
                        models['backbone_specular'] = PathNet(ic=n_in, outc=n_out)
                        p_in = 10 + n_out + 1 #23
                    else:
                        p_in = 47
                    models['dncnn'] = NewAdvKPCN_2(35, p_in, disc_activtion=args.disc_activation, output_type=args.output_type, strided_down=args.strided_down, interpolation=args.interpolation)
                else:
                    models['dncnn'] = AdvKPCN(n_in, pnet_out=pnet_out_size)
                    print('Initialize AdvKPCN for path descriptors (# of input channels: %d).'%(n_in))
                    n_in = dataset['train'].pnet_in_size
                    n_out = pnet_out_size
                    models['backbone_diffuse'] = PathNet(ic=n_in, outc=n_out)
                    models['backbone_specular'] = PathNet(ic=n_in, outc=n_out)
                    # models['KPN'] = KPN(ic=50)
            else:
                if not args.separate:
                    if args.manif_learn:
                        n_out = pnet_out_size
                        models['backbone'] = PathNet(ic=n_in, outc=n_out)
                        p_in = 20 + n_out + 1 #33
                    else:
                        p_in = 56
                    print("NewAdvKPCN_2 with gen activation:", args.activation, ", gen output type", args.output_type, "and disc activation", args.disc_activation)
                    # models['dncnn'] = NewAdvKPCN_2(45, 56, gen_activation=args.activation, disc_activtion=args.disc_activation, output_type=args.output_type, strided_down=args.strided_down, use_krn=args.use_krn)
                    models['dncnn'] = NewAdvKPCN_2(45, p_in, gen_activation=args.activation, disc_activtion=args.disc_activation, output_type=args.output_type, strided_down=args.strided_down, use_krn=args.use_krn)
                else:
                    print("separate and single with gen activation:", args.activation, ", gen output type", args.output_type, "and disc activation", args.disc_activation)
                    models['dncnn'] = ModKPCN(45, 56, activation=args.activation, output_type=args.output_type)
                    models['dis'] = PixelDiscriminator(64, 1, strided_down=args.strided_down, activation=args.disc_activation)
                    print("dncnn", models['dncnn'])
                    print("dis", models['dis'])
                print('Initialize AdvKPCN for path descriptors (# of input channels: %d).'%(n_in))
        else:
            n_in = dataset['train'].dncnn_in_size
            models['dncnn'] = AdvKPCN(n_in, pnet_out=pnet_out_size+2)
            print('Initialize AdvKPCN for vanilla buffers (# of input channels: %d).'%(n_in))
        
        # Load pretrained weights
        if len(list(itertools.product(*tmp))) == 1:
            model_fn = os.path.join(args.save, args.model_name + '.pth')
        else:
            model_fn = os.path.join(args.save, '%s_lp%f_pos%d_wgt%f.pth'%(args.model_name, lr_pnet, pnet_out_size, w_manif))
        assert args.start_epoch != 0 or not os.path.isfile(model_fn), 'Model %s already exists.'%(model_fn)
        is_pretrained = (args.start_epoch != 0) and os.path.isfile(model_fn)
        if is_pretrained:
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
            print('Pretraining weights are loaded.')
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
            lr = args.lr_dncnn if 'dncnn' == model_name else lr_pnet
            if args.use_pretrain and 'dncnn' == model_name:
                lr_finetune = 1e-6
                pre_params = []
                all_params = set(models[model_name].parameters())
                model_state_dict = models[model_name].state_dict()
                for k in model_state_dict:
                    if 'gbuf' in k: pre_params += list(model_state_dict[k])
                    # else: full_params += models[model_name][k]
                pre_params = set(pre_params)
                full_params = all_params - pre_params
                optims['optim_' + model_name] = optim.Adam(pre_params, lr=lr_finetune)
                optims['optim_' + model_name] = optim.Adam(full_params, lr=lr)
            else:
                optims['optim_' + model_name] = optim.Adam(models[model_name].parameters(), lr=lr)
            
            if not is_pretrained:
                continue

            if 'optims' in ck:
                state = ck['optims']['optim_' + model_name].state_dict()
            elif 'optim_' + model_name in ck['params']:
                state = ck['params']['optim_' + model_name].state_dict()
            else:
                print('No state for the optimizer for %s, use the initial optimizer and learning rate.'%(model_name))
                continue

            if not args.lr_ckpt:
                print('Set the new learning rate %.3e for %s.'%(lr, model_name))
                state['param_groups'][0]['lr'] = lr
            else:
                print('Use the checkpoint (%s) learning rate for %s.'%(model_fn, model_name))
            optims['optim_' + model_name].load_state_dict(state)

            # to remove error https://github.com/pytorch/pytorch/issues/80809
            optims['optim_' + model_name].param_groups[0]['capturable'] = True

        # Initialize losses (NOTE: modified for each model)
        if not args.soft_label:
            loss_funcs = {
                'l_diffuse': nn.L1Loss(),
                'l_specular': nn.L1Loss(),
                'l_recon': nn.L1Loss(),
                'l_test': RelativeMSE(),
                'l_adv': nn.BCELoss(),
            }
        else:
            if args.error_type == 'L1': l_adv = nn.L1Loss()
            elif args.error_type == 'MSE': l_adv = nn.MSELoss()
            else: l_adv = nn.L1Loss()
            loss_funcs = {
                'l_diffuse': nn.L1Loss(),
                'l_specular': nn.L1Loss(),
                'l_recon': nn.L1Loss(),
                'l_test': RelativeMSE(),
                'l_adv': l_adv
            }
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
        # print('interface', args.use_single, args.type=='new_adv_1')
        if not args.use_single:
            if args.type=='new_adv_1':
                itf = NewAdvKPCNInterface1(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, train_branches=args.train_branches, use_adv=args.use_adv)
            elif args.type == 'new_adv_2':
                itf = NewAdvKPCNInterface2(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, train_branches=args.train_branches, use_adv=args.use_adv)
            else:
                print('why?')
                itf = AdvKPCNInterface(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, use_adv=args.use_adv)
        else:
            if args.type == 'new_adv_1':
                itf = NewAdvKPCNInterface1(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, use_adv=args.use_adv)
            else:    
                itf = NewAdvKPCNInterface(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, use_adv=args.use_adv)
        # if is_pretrained:
        #     # print('Use the checkpoint best error %.3e'%(args.best_err))
        #     itf.best_err = args.best_err
        interfaces.append(itf)
    
    # Initialize a visdom visualizer object
    params = {
        'plots': {},
        'data_device': 1 if torch.cuda.device_count() > 1 and not args.single_gpu else args.device_id,
    }
    if args.visual:
        params['vis'] = visdom.Visdom(server='http://localhost')
    else:
        print('No visual.')
    
    # Make the save directory if needed
    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    return interfaces, params


def ddp_train(rank, args):
    # set up multi-processing
    torch.distributed.init_process_group("gloo", rank=rank, world_size=4) # world_size: numboer of processes
    # logger = logging.getLogger(__package__)
    # setup_logger()
    torch.cuda.set_device(rank)
    dataset, dataloaders = init_data(args)
    interfaces, params = init_model(dataset, args, rank)
    assert len(interfaces) > 1 or len(params) > 1

    ##### important!!!
    # make sure different processes sample different patches
    # np.random.seed((rank + 1) * 777)
    # random.seed((rank+1) * 777)
    # make sure different processes have different perturbations
    # torch.manual_seed((rank + 1) * 777)

    # only main process should log
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.summary, args.desc))
    
    for epoch in range(args.start_epoch, args.num_epoch):
        if len(interfaces) == 1:
            save_fn = args.model_name + '.pth'
        else:
            raise NotImplementedError('Multiple interfaces')

        start_time = time.time()

        # train
        for itf in interfaces:
            itf.to_train_mode()
        for i, batch in enumerate(dataloaders['train']):
            # Transfer data from the cpu to gpu memory
            for k in batch:
                if not batch[k].__class__ == torch.Tensor:
                    continue
                batch[k] = batch[k].cuda()


            # Main
            for itf in interfaces:
                itf.preprocess(batch, args.use_single)
                itf.train_batch(batch)
        
        summaries = []
        if not args.visual:
            for itf in interfaces:
                summaries.append(itf.get_epoch_summary(mode='train', norm=len(dataloaders['train'])))

        itf.epoch += 1
        itf.cnt = 0
        
        # only main process should log
        if rank == 0:
            logging_training(writer, epoch, 0, 1, summaries)
            print('[][] Elapsed time for one epoch: %d'%(time.time() - start_time))


        # save best model only on the main process
        if rank == 0:
            for i, itf in enumerate(interfaces):
                tmp_params = params.copy()
                tmp_params['vis'] = None

                state_dict = {
                    'description': args.desc, #
                    'start_epoch': epoch + 1,
                    'model': str(itf.models['dncnn']),
                    'params': tmp_params,
                    'optims': itf.optims,
                    'args': args,
                    'best_err': itf.best_err
                }

                for model_name in itf.models:
                    state_dict['state_dict_' + model_name] = itf.models[model_name].state_dict()

                if not args.not_save:
                    torch.save(state_dict, os.path.join(args.save, 'latest_' + save_fn))

        # validate
        if (epoch % args.val_epoch == args.val_epoch - 1):
            if rank == 0:
                print('[][] Validation')
            # summaries = validate_kpcn(epoch, interfaces, dataloaders, params, 0, args)
            print('[][] Validation (epoch %d split %d)' % (epoch, 0))
            for itf in interfaces:
                itf.to_eval_mode()

            summaries = []
            with torch.no_grad():
                for batch in tqdm(dataloaders['val'], leave=False, ncols=70):
                    # Transfer data from the cpu to gpu memory
                    for k in batch:
                        if not batch[k].__class__ == torch.Tensor:
                            continue
                        batch[k] = batch[k].cuda()

                    # Main
                    for itf in interfaces:
                        itf.validate_batch(batch)

            for itr in interfaces:
                summaries.append(itf.get_epoch_summary(mode='eval', norm=len(dataloaders['val'])))

            # save best model only on the main process
            if rank == 0:
                for i, itf in enumerate(interfaces):
                    if summaries[i] < itf.best_err:
                        itf.best_err = summaries[i]

                        tmp_params = params.copy()
                        tmp_params['vis'] = None

                        state_dict = {
                            'description': args.desc, #
                            'start_epoch': epoch + 1,
                            'model': str(itf.models['dncnn']),
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

                    print('[][] Model {} RelMSE: {:.3f}e-3 \t Best RelMSE: {:.3f}e-3'.format(save_fn, summaries[i]*1000, itf.best_err*1000))
                    logging(writer, epoch, 0, 1, summaries[i], itf.best_err)

        # # Update schedulers
        for key in params:
            if 'sched_' in key:
                params[key].step()

    torch.distributed.destroy_process_group()


def main(args):
    # Set random seeds
    random.seed("Inyoung Cho, Yuchi Huo, Sungeui Yoon @ KAIST")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True  #torch.backends.cudnn.deterministic = True

    # Get ready
    if not args.distributed:
        dataset, dataloaders = init_data(args)
        interfaces, params = init_model(dataset, args)
        train(interfaces, dataloaders, params, args)
    else:
        print('gpu count', torch.cuda.device_count())
        mp.spawn(ddp_train,
                 args=(args,),
                 nprocs=torch.cuda.device_count(),
                 join=True)


if __name__ == "__main__":
    """ NOTE: Example Training Scripts """
    """ KPCN Vanilla
        Train two branches (i.e., diffuse and specular):
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_vanilla --desc "KPCN vanilla" --num_epoch 8 --lr_dncnn 1e-4 --train_branches

        Post-joint training ('fine-tuning' according to the original authors): 
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_vanilla --desc "KPCN vanilla" --num_epoch 10 --lr_dncnn 1e-6 --start_epoch ?
    """

    """ KPCN Manifold
        Train two branches (i.e., diffuse and specular):
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_manifold_FMSE --desc "KPCN manifold FMSE" --num_epoch 8 --manif_loss FMSE --lr_dncnn 1e-4 --lr_pnet 1e-4 --use_llpm_buf --manif_learn --w_manif 0.1 --train_branches

        Post-joint training:
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_manifold_FMSE --desc "KPCN manifold FMSE" --num_epoch 10 --manif_loss FMSE --lr_dncnn 1e-6 --lr_pnet 1e-6 --use_llpm_buf --manif_learn --w_manif 0.1 --start_epoch <best pre-training epoch>
    """

    """ KPCN Path (ablation study)
        Train two branches (i.e., diffuse and specular):
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_path --desc "KPCN ablation study" --num_epoch 8 --lr_dncnn 1e-4 --lr_pnet 1e-4 --use_llpm_buf --train_branches

        Post-joint training:
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_path --desc "KPCN ablation study" --num_epoch 10 --lr_dncnn 1e-6 --lr_pnet 1e-6 --use_llpm_buf --start_epoch <best pre-training epoch>
    """

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

    parser.add_argument('--kpcn_ref', action='store_true',
                        help='train KPCN-Ref model.')
    parser.add_argument('--kpcn_pre', action='store_true',
                        help='train KPCN-Pre model.')
    parser.add_argument('--not_save', action='store_true',
                        help='do not save checkpoint (debugging purpose).')
    parser.add_argument('--local', action='store_true', help='device id')

    # new arguments
    parser.add_argument('--use_skip', action='store_true')
    parser.add_argument('--use_adv', action='store_true')
    parser.add_argument('--use_pretrain', action='store_true')
    parser.add_argument('--w_adv', nargs='+', default=[0.0001], 
                        help='ratio of the adversarial learning loss to \
                        the reconstruction loss.')
    parser.add_argument('--use_single', action='store_true')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--strided_down', action='store_true')
    parser.add_argument('--soft', action='store_true')

    # (for NewAdvKPCN_2)
    parser.add_argument('--activation', type=str, default='relu',
                        help='`relu`, `leaky_relu`, `tanh`, or `elu`')
    parser.add_argument('--output_type', type=str, default='linear',
                        help='`linear`, `relu`, `leaky_relu`, `sigmoid`, `tanh`, `elu`, or `softplus`,')
    parser.add_argument('--disc_activation', type=str, default='relu',
                        help='`relu`, `leaky_relu`, or `linear`')
    parser.add_argument('--use_krn', action='store_true')

    # (for NewAdvKPCN_1)
    parser.add_argument('--type', type=str, default='new_adv_2')
    parser.add_argument('--revise', action='store_true')
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--error_type', type=str, default='L1')
    parser.add_argument('--interpolation', type=str, default='kernel')
    parser.add_argument('--weight', type=str, default='sigmoid')
    
    
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
