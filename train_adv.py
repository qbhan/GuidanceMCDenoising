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
from support.networks import AdvKPCN, NewAdvKPCN_2, ModKPCN, PixelDiscriminator, NewAdvKPCN_1
from support.datasets import MSDenoiseDataset, DenoiseDataset
from support.utils import BasicArgumentParser
from support.losses import RelativeMSE, FeatureMSE, GlobalRelativeSimilarityLoss
from support.interfaces import AdvKPCNInterface, NewAdvKPCNInterface, NewAdvKPCNInterface1
from train_kpcn import validate_kpcn, train, train_epoch_kpcn

# Gharbi et al. dependency
sys.path.insert(1, configs.PATH_SBMC)
try:
    from sbmc import KPCN
except ImportError as error:
    print('Put appropriate paths in the configs.py file.')
    raise


def init_data(args):
    # Initialize datasets
    datasets = {}
#    datasets['train'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
#        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
#    datasets['val'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
#        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
    datasets['train'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
         use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
    datasets['val'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
         use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3, use_single=args.use_single)
    
    # Initialize dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets['train'], 
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=False,
    )
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=BS_VAL,
        num_workers=1,
        pin_memory=False
    )
    return datasets, dataloaders


def init_model(dataset, args):
    interfaces = []

    lr_pnets = args.lr_pnet
    pnet_out_sizes = args.pnet_out_size
    w_manifs = args.w_manif

    tmp = [lr_pnets, pnet_out_sizes, w_manifs]
    for lr_pnet, pnet_out_size, w_manif in list(itertools.product(*tmp)):
        # Initialize models (NOTE: modified for each model) 
        models = {}
            
        if args.use_llpm_buf:
            if args.disentangle in ['m10r01', 'm11r01']:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size // 2
            else:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size
            print('adv', n_in, pnet_out_size)
            if not args.use_single:
                print(args.type)
                print(args.type == 'new_adv_1')
                if args.type == 'new_adv_1':
                    if args.manif_learn:
                        n_in = dataset['train'].pnet_in_size
                        n_out = pnet_out_size
                        models['backbone_diffuse'] = PathNet(ic=n_in, outc=n_out)
                        models['backbone_specular'] = PathNet(ic=n_in, outc=n_out)
                        p_in = 10 + n_out + 1 #23
                    else:
                        p_in = 46
                    models['dncnn'] = NewAdvKPCN_1(35, p_in, gen_activation=args.activation, disc_activtion=args.disc_activation, output_type=args.output_type, strided_down=args.strided_down)
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
            print('Data Parallel')
            if torch.cuda.device_count() == 1:
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

        # Initialize losses (NOTE: modified for each model)
        loss_funcs = {
            'l_diffuse': nn.L1Loss(),
            'l_specular': nn.L1Loss(),
            'l_recon': nn.L1Loss(),
            'l_test': RelativeMSE(),
            'l_adv': nn.BCELoss(),
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

        # Initialize a training interface (NOTE: modified for each model)
        if not args.use_single:
            if args.type == 'new_adv_1':
                itf = NewAdvKPCNInterface1(models, optims, loss_funcs, args, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, use_adv=args.use_adv)
            else:
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
    print('device:', args.device_id)
    torch.cuda.set_device(f'cuda:{args.device_id}')
    main(args)
