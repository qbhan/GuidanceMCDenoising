"""Encapsulation of model training and testing for learning light path manifold.
"""

# Python
import sys
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
# NumPy and PyTorch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# Gharbi et al. dependency
#
# Cho et al. dependency
from support.utils import crop_like
from torchvision.utils import save_image
import os

import configs
sys.path.insert(1, configs.PATH_SBMC)
try:
    from sbmc import modules as ops
except ImportError as error:
    print('Put appropriate paths in the configs.py file.')
    raise

from support.utils import ToneMapBatch

def score_map(img, tgt, error_type='L1', Tonemapping=False):
    # with torch.no_grad():
    #     score = torch.clip(torch.abs((ToneMapBatch(img) - ToneMapBatch(tgt))), 0.0, 1.0)
    #     return score
    eps = 0.000001
    # print('score type', error_type)
    if Tonemapping:
        img, tgt = ToneMapBatch(img), ToneMapBatch(tgt)
    if error_type == 'L1':
        score = torch.abs((img - tgt)).mean(dim=1)
    elif error_type == 'MSE':
        # print(img.shape, tgt.shape)
        score = torch.square((img - tgt)).mean(dim=1)
    elif error_type == 'relMSE':
        mse = torch.square((img - tgt))
        score = mse / (torch.square(tgt) + eps)
        score = score.mean(1)
    elif error_type == 'relL1':
        l1 = torch.abs((img - tgt))
        score = l1 / (torch.abs(tgt) + eps)
        score = score.mean(1)
    elif error_type == 'SMAPE':
        score = (torch.abs((img - tgt))) / (
            eps + torch.abs(img) + torch.abs(tgt)
        )
        score = score.mean(1)
        # print(score.shape)
    score = torch.clip(score, 0.0, None) 
    # score = torch.nn.
    score = score.unsqueeze(1)
    return score.detach()


  

class BaseInterface(metaclass=ABCMeta):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1):
        self.models = models
        self.optims = optims
        self.loss_funcs = loss_funcs
        self.args = args
        self.visual = visual
        self.use_llpm_buf = use_llpm_buf
        self.manif_learn = manif_learn
        self.w_manif = w_manif

        self.iters = 0
        self.m_losses = {}
        self.best_err = 1e10
        self.fixed_batch = None

    @abstractmethod
    def to_train_mode(self):
        pass

    @abstractmethod
    def preprocess(self, batch=None):
        pass

    @abstractmethod
    def train_batch(self, batch):
        pass

    @abstractmethod
    def _manifold_forward(self, batch):
        return {}

    @abstractmethod
    def _regress_forward(self, batch):
        return {}

    @abstractmethod
    def _backward(self, batch, out, p_buffers):
        return {}
    
    @abstractmethod
    def _logging(self, loss_dict):
        pass

    @abstractmethod
    def _optimization(self):
        pass

    @abstractmethod
    def to_eval_mode(self):
        pass

    @abstractmethod
    def validate_batch(self, batch):
        pass

    @abstractmethod
    def get_epoch_summary(self, mode, norm):
        return 0.0


class KPCNInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_pretrain=False, apply_loss_twice=False):
        if manif_learn:
            assert 'backbone_diffuse' in models, "argument `models` dictionary should contain `'backbone_diffuse'` key."
            assert 'backbone_specular' in models, "argument `models` dictionary should contain `'backbone_specular'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentanglement_option in ['m11r11', 'm10r01', 'm11r01', 'm10r11']
        
        super(KPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.use_skip = use_skip
        self.use_pretrain = use_pretrain
        self.apply_loss_twice = apply_loss_twice
        self.cnt = 0
        self.epoch = 0
        self.no_p_model = args.no_p_model
        self.no_gbuf = args.no_gbuf

    def __str__(self):
        return 'KPCNInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None, use_single=False):
        assert 'target_total' in batch
        assert 'target_diffuse' in batch
        assert 'target_specular' in batch
        assert 'kpcn_diffuse_in' in batch
        assert 'kpcn_specular_in' in batch
        assert 'kpcn_diffuse_buffer' in batch
        assert 'kpcn_specular_buffer' in batch
        assert 'kpcn_albedo' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()
            p_buffers = self._manifold_forward(batch)

            # if self.iters % 1000 == 1:
            #     pimg = np.mean(np.transpose(p_buffers['diffuse'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s_diffuse.png'%(self.args.model_name), pimg)

            #     pimg = np.mean(np.transpose(p_buffers['specular'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s_specular.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm11r11':
                out_manif = p_buffers
            elif self.disentanglement_option == 'm10r01':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm11r01':
                out_manif = p_buffers
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm10r11':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            if not self.no_gbuf:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }
            else:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'][:, :10], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'][:, :10], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }

            # del p_buffer
        
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)
        del loss_dict
        self._optimization()

    def _manifold_forward(self, batch):
        if not self.no_p_model:
            p_buffer_diffuse = self.models['backbone_diffuse'](batch)
            p_buffer_specular = self.models['backbone_specular'](batch)
            p_buffers = {
                'diffuse': p_buffer_diffuse,
                'specular': p_buffer_specular
            }
        else:
            p_buffers = {
                'diffuse': batch['paths'],
                'specular': batch['paths']
            }
        return p_buffers

    def _regress_forward(self, batch):
        return self.models['dncnn'](batch)

    def _backward(self, batch, out, p_buffers):
        assert 'radiance' in out
        assert 'diffuse' in out
        assert 'specular' in out

        total, diffuse, specular = out['radiance'], out['diffuse'], out['specular']
        if self.use_skip:
            gbuf_total, gbuf_diffuse, gbuf_specular = out['gbuf_radiance'], out['gbuf_diffuse'], out['gbuf_specular']
        loss_dict = {}
        tgt_total = crop_like(batch['target_total'], total)
        if self.train_branches: # training diffuse and specular branches
            tgt_diffuse = crop_like(batch['target_diffuse'], diffuse)
            L_diffuse = self.loss_funcs['l_diffuse'](diffuse, tgt_diffuse)

            tgt_specular = crop_like(batch['target_specular'], specular)
            L_specular = self.loss_funcs['l_specular'](specular, tgt_specular)

            loss_dict['l_diffuse'] = L_diffuse.detach()
            loss_dict['l_specular'] = L_specular.detach()

            # joint training refinement & denoising
            if self.use_skip and not self.use_pretrain:
                L_gbuf_diffuse = self.loss_funcs['l_diffuse'](gbuf_diffuse, tgt_diffuse)

                L_gbuf_specular = self.loss_funcs['l_specular'](gbuf_specular, tgt_specular)

                loss_dict['l_gbuf_diffuse'] = L_gbuf_diffuse.detach()
                loss_dict['l_gbuf_specular'] = L_gbuf_specular.detach()
    
                L_diffuse += L_gbuf_diffuse
                L_specular += L_gbuf_specular
            
            if self.apply_loss_twice:
                # backward with kernel only from g-buffer
                diffuse_g, specular_g = out['gbuf_diffuse'], out['gbuf_specular']
                # tgt_diffuse_p = crop_like(batch['target_diffuse'], diffuse_g)
                L_diffuse_p = self.loss_funcs['l_diffuse'](diffuse_g, tgt_diffuse)

                # tgt_specular_p = crop_like(batch['target_specular'], specular_g)
                L_specular_p = self.loss_funcs['l_specular'](specular_g, tgt_specular)

                loss_dict['l_diffuse_p'] = L_diffuse_p.detach()
                loss_dict['l_specular_p'] = L_specular_p.detach()

            if self.manif_learn:
                p_buffer_diffuse = crop_like(p_buffers['diffuse'], diffuse)
                L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse)
                L_diffuse += L_manif_diffuse * self.w_manif

                p_buffer_specular = crop_like(p_buffers['specular'], specular)
                L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular)
                L_specular += L_manif_specular * self.w_manif

                loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach()
                loss_dict['l_manif_specular'] = L_manif_specular.detach()

                del p_buffer_diffuse
                del p_buffer_specular
            
            L_diffuse.backward()
            L_specular.backward()

            with torch.no_grad():
                L_total = self.loss_funcs['l_recon'](total, tgt_total)
                
                if self.apply_loss_twice:
                    total_g = out['gbuf_radiance']
                    tgt_total_g = crop_like(batch['target_total'], total_g)
                    L_total_g = self.loss_funcs['l_recon'](total_g, tgt_total_g)
                    loss_dict['l_total_g'] = L_total_g.detach()
                
                loss_dict['l_total'] = L_total.detach()
        else: # post-training the entire system
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            if self.apply_loss_twice:
                    total_g = out['radiance_p']
                    tgt_total_g = crop_like(batch['target_total'], total_g)
                    L_total += self.loss_funcs['l_recon'](total_g, tgt_total_g)
            loss_dict['l_total'] = L_total.detach()
            L_total.backward()
        
        del diffuse
        del specular  
        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](total, tgt_total).detach()
        self.cnt += 1
        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        for model_name in self.models:
            nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            self.optims['optim_' + model_name].step()

    def to_eval_mode(self):
        for model_name in self.models:
            self.models[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)
        self.m_losses['m_gbuf_val'] = torch.tensor(0.0)

    def validate_batch(self, batch, mode=None):
        p_buffers = None

        if self.use_llpm_buf:
            p_buffers = self._manifold_forward(batch)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm10r01' or self.disentanglement_option == 'm11r01':
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            if not self.no_gbuf:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }
            else:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'][:, :10], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'][:, :10], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }
        
        out = self._regress_forward(batch)
        rad_dict = {'diffuse': out['diffuse'],
                    'specular': out['specular']}
        kernel_dict = {'diffuse': None,
                      'specular': None}
        
        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        if self.use_skip and not self.use_pretrain:
            rad_dict['g_radiance'] = out['gbuf_radiance']
            rad_dict['g_diffuse'] = out['gbuf_diffuse']
            rad_dict['g_specular'] = out['gbuf_specular']
            kernel_dict['g_diffuse'] = out['k_g_diffuse']
            kernel_dict['g_specular'] = out['k_g_specular']
            L_gbuf_total = self.loss_funcs['l_test'](out['gbuf_radiance'], tgt_total)
            if self.m_losses['m_gbuf_val'] == 0.0 and self.m_losses['m_gbuf_val'].device != L_gbuf_total.device:
                self.m_losses['m_gbuf_val'] = torch.tensor(0.0, device=L_gbuf_total.device)
            L_gbuf_total = self.loss_funcs['l_test'](out['gbuf_radiance'], tgt_total)
            self.m_losses['m_gbuf_val'] += L_gbuf_total.detach()


        return out['radiance'],  p_buffers, rad_dict, kernel_dict, None

    def get_epoch_summary(self, mode, norm):
        if mode == 'train':
            losses = {}
            print('[][][]', end=' ')
            for key in self.m_losses:
                if key == 'm_val':
                    continue
                tr_l_tmp = self.m_losses[key] / (norm * 2)
                tr_l_tmp *= 1000
                print('%s: %.3fE-3'%(key, tr_l_tmp), end='\t')
                losses[key] = tr_l_tmp
                self.m_losses[key] = torch.tensor(0.0, device=self.m_losses[key].device)
            print('')
            return losses #-1.0
        else:
            return self.m_losses['m_val'].item() / (norm * 2)


class AdvMCDInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_pretrain=False, apply_loss_twice=False):
        if manif_learn:
            assert 'backbone_diffuse' in models, "argument `models` dictionary should contain `'backbone_diffuse'` key."
            assert 'backbone_specular' in models, "argument `models` dictionary should contain `'backbone_specular'` key."
        assert 'generator_diffuse' in models, "argument `models` dictionary should contain `'dncnn'` key."
        assert 'generator_specular' in models, "argument `models` dictionary should contain `'dncnn'` key."
        assert 'discriminator_diffuse' in models, "argument `models` dictionary should contain `'discriminator'` key."
        assert 'discriminator_specular' in models, "argument `models` dictionary should contain `'discriminator'` key."
        assert 'l_diffuse' in loss_funcs
        assert 'l_specular' in loss_funcs
        assert 'l_gan' in loss_funcs
        assert 'l_gp' in loss_funcs
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentanglement_option in ['m11r11', 'm10r01', 'm11r01', 'm10r11']
        
        super(AdvMCDInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.disentanglement_option = disentanglement_option
        self.cnt = 0
        self.epoch = 0
        # self.no_gbuf = args.no_gbuf
        self.D_update_ratio = 1
        self.D_init_iters = 10
        self.gan_weight = 5e-3
        self.gp_weight = 10
        self.step = args.start_epoch * 14400 #hardcoded
        self.random_pt = torch.Tensor(1, 1, 1, 1).cuda()
        

    def __str__(self):
        return 'AdvMCDInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None, use_single=False):
        assert 'target_total' in batch
        assert 'target_diffuse' in batch
        assert 'target_specular' in batch
        assert 'kpcn_diffuse_in' in batch
        assert 'kpcn_specular_in' in batch
        assert 'kpcn_diffuse_buffer' in batch
        assert 'kpcn_specular_buffer' in batch
        assert 'kpcn_albedo' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()
            p_buffers = self._manifold_forward(batch)


            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm11r11':
                out_manif = p_buffers
            elif self.disentanglement_option == 'm10r01':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm11r01':
                out_manif = p_buffers
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm10r11':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            if not self.no_gbuf:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }
            else:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'][:, :10], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'][:, :10], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }            

        gt_total, gt_diff, gt_spec = batch['target_total'], batch['target_diffuse'], batch['target_specular']
        loss_dict = {}

        # optimize Generators (and Path-manifold module)
        self.models['generator_diffuse'].zero_grad()
        self.models['generator_specular'].zero_grad()
        out_fake = self._regress_forward(batch)
        l_diff_total, l_spec_total = 0.0, 0.0

        # recon loss
        l_diff_recon = self.loss_funcs['l_diffuse'](out_fake['diffuse'], gt_diff)
        l_spec_recon = self.loss_funcs['l_specular'](out_fake['specular'], gt_spec)
        loss_dict['l_G_diffuse_recon'], loss_dict['l_G_specular_recon'] = l_diff_recon.detach(), l_spec_recon.detach()
        l_diff_total += l_diff_recon
        l_spec_total += l_spec_recon

        # fake discriminator loss
        pred_fake = self._discriminator_forward(out_fake)
        l_diff_gan = self.loss_funcs['l_gan'](pred_fake['diffuse'], True)
        l_spec_gan = self.loss_funcs['l_gan'](pred_fake['specular'], True)
        loss_dict['l_G_diffuse_gan_fake'], loss_dict['l_G_specular_gan_fake'] = l_diff_gan.detach(), l_spec_gan.detach()
        l_diff_total += l_diff_gan * self.gan_weight
        l_spec_total += l_spec_gan * self.gan_weight

        # path manifold loss
        if self.manif_learn:
            p_buffer_diffuse = crop_like(p_buffers['diffuse'], out_fake['diffuse'])
            L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, gt_diff)
            L_diff_total += L_manif_diffuse * self.w_manif

            p_buffer_specular = crop_like(p_buffers['specular'], out_fake['specular'])
            L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, gt_spec)
            L_spec_total += L_manif_specular * self.w_manif

            loss_dict['l_G_manif_diffuse'] = L_manif_diffuse.detach()
            loss_dict['l_G_manif_specular'] = L_manif_specular.detach()

        loss_dict['l_G_diffuse_total'] = l_diff_total.detach()
        loss_dict['l_G_specular_total'] = l_spec_total.detach()

        l_diff_total.backward()
        l_spec_total.backward()
        self.optims['optim_generator_diffuse'].step()
        self.optims['optim_generator_specular'].step()
        if self.manif_learn:
            self.optims['optim_backbone_diffuse'].step()
            self.optims['optim_backbone_specular'].step()
        
        
        # optimize discriminators
        self.models['discriminator_diffuse'].zero_grad()
        self.models['discriminator_specular'].zero_grad()
        l_diff_total, l_spec_total = 0.0, 0.0
        real_data = dict(diffuse=gt_diff, specular=gt_spec)
        pred_real = self._discriminator_forward(real_data)
        l_diff_gan_real = self.loss_funcs['l_gan'](pred_real['diffuse'], True)
        l_spec_gan_real = self.loss_funcs['l_gan'](pred_real['specular'], True)
        fake_data = dict(diffuse=out_fake['diffuse'].detach(), specular=out_fake['specular'].detach())
        pred_fake = self._discriminator_forward(fake_data)
        l_diff_gan_fake = self.loss_funcs['l_gan'](pred_fake['diffuse'], False)
        l_spec_gan_fake = self.loss_funcs['l_gan'](pred_fake['specular'], False)
        loss_dict['l_D_diffuse_gan_real'] = l_diff_gan_real.detach()
        loss_dict['l_D_specular_gan_real'] = l_spec_gan_real.detach()
        loss_dict['l_D_diffuse_gan_fake'] = l_diff_gan_fake.detach()
        loss_dict['l_D_specular_gan_fake'] = l_spec_gan_fake.detach()
        l_diff_total += (l_diff_gan_fake + l_diff_gan_real) / 2.0
        l_spec_total += (l_spec_gan_fake + l_spec_gan_real) / 2.0

        # self.random_pt_spec = torch.Tensor.cuda((1, 1, 1, 1))
        B = out_fake['diffuse'].shape[0]
        if self.random_pt.shape[0] != B:
            self.random_pt.resize_(B, 1, 1, 1)
        self.random_pt.uniform_()
        interp_diff = self.random_pt * fake_data['diffuse'] + (1 - self.random_pt) * gt_diff
        interp_diff.requires_grad = True
        self.random_pt.uniform_()
        interp_spec = self.random_pt * fake_data['specular'] + (1 - self.random_pt) * gt_spec
        interp_spec.requires_grad = True
        interp_in = dict(diffuse=interp_diff, specular=interp_spec)
        interp_crit = self._discriminator_forward(interp_in)
        l_diff_gan_gp = self.loss_funcs['l_gp'](interp_diff, interp_crit['diffuse'])
        l_spec_gan_gp = self.loss_funcs['l_gp'](interp_spec, interp_crit['specular'])
        loss_dict['l_D_diffuse_gan_gp'] = l_diff_gan_gp.detach()
        loss_dict['l_D_specular_gan_gp'] = l_spec_gan_gp.detach()
        l_diff_total += l_diff_gan_gp * self.gp_weight
        l_spec_total += l_spec_gan_gp * self.gp_weight

        loss_dict['l_D_diffuse_total'] = l_diff_total.detach()
        loss_dict['l_D_specular_total'] = l_spec_total.detach()
        l_diff_total.backward()
        l_spec_total.backward()
        self.optims['optim_discriminator_diffuse'].step()
        self.optims['optim_discriminator_specular'].step()

        # final recon loss for log
        albedo = crop_like(batch['kpcn_albedo'], out_fake['diffuse'])
        final_recon = out_fake['diffuse'] * albedo + torch.exp(out_fake['specular']) - 1.0
        with torch.no_grad():
            L_total = self.loss_funcs['l_recon'](final_recon, gt_total)
        loss_dict['l_total'] = L_total.detach()
        
        # log all losses in this step
        self._logging(loss_dict)

    def _manifold_forward(self, batch):
        if not self.no_p_model:
            p_buffer_diffuse = self.models['backbone_diffuse'](batch)
            p_buffer_specular = self.models['backbone_specular'](batch)
            p_buffers = {
                'diffuse': p_buffer_diffuse,
                'specular': p_buffer_specular
            }
        else:
            p_buffers = {
                'diffuse': batch['paths'],
                'specular': batch['paths']
            }
        return p_buffers


    def _discriminator_forward(self, data):
        return dict(
            diffuse=self.models['discriminator_diffuse'](data['diffuse']), 
            specular=self.models['discriminator_specular'](data['specular'])
            )

    def _regress_forward(self, batch):
        r_diff = batch['kpcn_diffuse_buffer']
        f_diff = torch.cat((batch['kpcn_diffuse_in'][:, 10:13], batch['kpcn_diffuse_in'][:, 20:21], batch['kpcn_diffuse_in'][:, 24:27]), dim=1)
        r_spec = batch['kpcn_specular_buffer']
        f_spec = torch.cat((batch['kpcn_specular_in'][:, 10:13], batch['kpcn_specular_in'][:, 20:21], batch['kpcn_specular_in'][:, 24:27]), dim=1)

        # denoise
        d_diff = self.models['generator_diffuse']((r_diff, f_diff))
        d_spec = self.models['generator_diffuse']((r_spec, f_spec))
        return dict(diffuse=d_diff, specular=d_spec)


    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        for model_name in self.models:
            nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    

    def to_eval_mode(self):
        for model_name in self.models:
            self.models[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)
        self.m_losses['m_gbuf_val'] = torch.tensor(0.0)

    def validate_batch(self, batch, mode=None):
        p_buffers = None

        if self.use_llpm_buf:
            p_buffers = self._manifold_forward(batch)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm10r01' or self.disentanglement_option == 'm11r01':
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            if not self.no_gbuf:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }
            else:
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'][:, :10], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'][:, :10], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                }

        out = self._regress_forward(batch)
        albedo = crop_like(batch['kpcn_albedo'], out['diffuse'])
        final_recon = out['diffuse'] * albedo + torch.exp(out['specular']) - 1.0
        rad_dict = {
                    'diffuse': out['diffuse'],
                    'specular': out['specular']}
        kernel_dict = {'diffuse': None,
                      'specular': None}
        
        tgt_total = crop_like(batch['target_total'], final_recon)
        L_total = self.loss_funcs['l_test'](final_recon, tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return final_recon,  p_buffers, rad_dict, kernel_dict, None

    def get_epoch_summary(self, mode, norm):
        if mode == 'train':
            losses = {}
            print('[][][]', end=' ')
            for key in self.m_losses:
                if key == 'm_val':
                    continue
                tr_l_tmp = self.m_losses[key] / (norm * 2)
                tr_l_tmp *= 1000
                print('%s: %.3fE-3'%(key, tr_l_tmp), end='\t')
                losses[key] = tr_l_tmp
                self.m_losses[key] = torch.tensor(0.0, device=self.m_losses[key].device)
            print('')
            return losses #-1.0
        else:
            return self.m_losses['m_val'].item() / (norm * 2)

    def _backward():
        pass

    def _optimization():
        pass

class SBMCInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, use_sbmc_buf=True, disentangle="m11r11"):
        if manif_learn:
            assert 'backbone' in models, "argument `models` dictionary should contain `'backbone'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentangle in ['m11r11', 'm10r01', 'm11r01', 'm10r11']

        super(SBMCInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.disentangle = disentangle
        self.use_sbmc_buf = use_sbmc_buf
        self.cnt = 0
        self.epoch = 0
    
    def __str__(self):
        return 'SBMCInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None, use_single=False):
        assert 'target_image' in batch
        assert 'radiance' in batch
        assert 'features' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch
        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone'].zero_grad()
            p_buffer = self._manifold_forward(batch)

            # if self.iters % 1000 == 1:
            #     pimg = np.mean(np.transpose(p_buffer.detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, s, c, _, _ = p_buffer.shape
            assert c >= 2
            if self.disentangle == 'm11r11':
                out_manif = p_buffer
            elif self.disentangle == 'm10r01':
                out_manif = p_buffer[:,:,c//2:,...]
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm11r01':
                out_manif = p_buffer
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm10r11':
                out_manif = p_buffer[:,:,c//2:,...]

            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= s # spp
            p_var = torch.stack([p_var,]*s, axis=1).detach()

            # make a new batch
            batch = {
                'target_image': batch['target_image'],
                'radiance': batch['radiance'],
                'features':  torch.cat([batch['features'], p_buffer, p_var], 2),
            }

        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        self._optimization()
        
    def _manifold_forward(self, batch):
        return self.models['backbone'](batch)

    def _regress_forward(self, batch):
        return self.models['dncnn'](batch)
    
    def _backward(self, batch, out, p_buffer):
        loss_dict = {}
        tgt_total = crop_like(batch['target_image'], out)

        L_total = self.loss_funcs['l_recon'](out, tgt_total)

        if self.manif_learn:
            p_buffer = crop_like(p_buffer, out)
            L_manif = self.loss_funcs['l_manif'](p_buffer, tgt_total)

            loss_dict['l_manif'] = L_manif.detach()
            loss_dict['l_recon'] = L_total.detach()

            L_total += L_manif * self.w_manif
        
        loss_dict['l_total'] = L_total.detach()

        L_total.backward()

        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](out, tgt_total).detach()

        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        clip = 1000
        for model_name in self.models:
            actual = nn.utils.clip_grad_norm_(self.models[model_name].parameters(), max_norm=clip)
            if actual > clip:
                print("Clipped %s gradients %f -> %f"%(model_name, clip, actual))

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            self.optims['optim_' + model_name].step()

    def to_eval_mode(self):
        for model_name in self.models:
            self.models[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)
    
    def validate_batch(self, batch, mode=False):
        p_buffer = None

        if self.use_llpm_buf:
            p_buffer = self._manifold_forward(batch)
            
            """ Feature disentanglement """
            _, s, c, _, _ = p_buffer.shape
            assert c >= 2
            if self.disentangle == 'm10r01':
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm11r01':
                p_buffer = p_buffer[:,:,:c//2,...]

            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= s # spp
            p_var = torch.stack([p_var,]*s, axis=1).detach()

            # make a new batch
            batch = {
                'target_image': batch['target_image'],
                'radiance': batch['radiance'],
                'features':  torch.cat([batch['features'], p_buffer, p_var], 2),
            }
        
        out = self._regress_forward(batch)

        tgt_total = crop_like(batch['target_image'], out)
        L_total = self.loss_funcs['l_test'](out, tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out, p_buffer, dict(), dict(), dict()
    
    def get_epoch_summary(self, mode, norm):
        if mode == 'train':
            print('[][][]', end=' ')
            losses = {}
            for key in self.m_losses:
                if key == 'm_val':
                    continue
                tr_l_tmp = self.m_losses[key] / (norm * 2)
                tr_l_tmp *= 1000
                print('%s: %.3fE-3'%(key, tr_l_tmp), end='\t')
                losses[key] = tr_l_tmp
                self.m_losses[key] = torch.tensor(0.0, device=self.m_losses[key].device)
            print('')
            return losses #-1.0
        else:
            return self.m_losses['m_val'].item() / (norm * 2)


class LBMCInterface(SBMCInterface):

    def __init__(self, models, optims, loss_funcs, args, use_llpm_buf=False, manif_learn=False, w_manif=0.1, disentangle='m11r11'):
        if manif_learn:
            assert 'backbone' in models, "argument `models` dictionary should contain `'backbone'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentangle in ['m11r11', 'm10r01', 'm11r01', 'm10r11']

        super(LBMCInterface, self).__init__(models, optims, loss_funcs, args, False, use_llpm_buf, manif_learn, w_manif, False, disentangle)
    
    def __str__(self):
        return 'LBMCInterface'
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone'].zero_grad()
            p_buffer = self._manifold_forward(batch)

            # if self.iters % 1000 == 1:
            #     pimg = np.mean(np.transpose(p_buffer.detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, s, c, _, _ = p_buffer.shape
            assert c >= 2
            if self.disentangle == 'm11r11':
                out_manif = p_buffer
            elif self.disentangle == 'm10r01':
                out_manif = p_buffer[:,:,c//2:,...]
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm11r01':
                out_manif = p_buffer
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm10r11':
                out_manif = p_buffer[:,:,c//2:,...]

            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= s # spp
            p_var = torch.stack([p_var,]*s, axis=1).detach()

            # make a new batch
            batch = {
                'target_image': batch['target_image'],
                'radiance': batch['radiance'],
                'features':  torch.cat([batch['features'], p_buffer, p_var], 2),
            }
        
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        if not self.amp:
            self._optimization()
    
    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        GRADIENT_CLAMP_N = 0.25 * 1000
        #GRADIENT_CLAMP   = 0.001 * 1000
        for model_name in self.models:
            actual = nn.utils.clip_grad_norm_(self.models[model_name].parameters(), GRADIENT_CLAMP_N)
            #nn.utils.clip_grad_value_(self.models[model_name].parameters(), GRADIENT_CLAMP)
            if actual > GRADIENT_CLAMP_N:
                print("Clipped %s gradients %f -> %f"%(model_name, GRADIENT_CLAMP_N, actual))
                #self.models[model_name].zero_grad()

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]


class EnsembleKPCNInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", w_error=1.0, error_type='L1'):
        if manif_learn:
            assert 'backbone_diffuse' in models, "argument `models` dictionary should contain `'backbone_diffuse'` key."
            assert 'backbone_specular' in models, "argument `models` dictionary should contain `'backbone_specular'` key."
        assert 'dncnn_G' in models, "argument `models` dictionary should contain `'dncnn_G'` key."
        assert 'dncnn_P' in models, "argument `models` dictionary should contain `'dncnn_P'` key."
        # assert 'interpolate' in models, "argument `models` dictionary should contain `interpolation_diffuse` key."
        # assert 'interpolate_specular' in models, "argument `models` dictionary should contain `interpolation_specular` key."
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentanglement_option in ['m11r11', 'm10r01', 'm11r01', 'm10r11']
        
        super(EnsembleKPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.fix = args.fix
        self.feature = args.feature
        self.model_type = args.model_type
        self.best_err = 1e10
        self.epoch = 0
        self.cnt = 0    
        # self.weight = 1.0
        self.weight = args.weight
        self.schedule = args.schedule
        self.ensemble_branches = args.ensemble_branches
        self.train_branches = self.ensemble_branches

        # print('ErrorEnsembleKPCN')
        # print('interpolate', self.interpolate)
        
    def __str__(self):
        return 'EnsembleKPCN'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            if self.fix and ('dncnn' in model_name or 'backbone' in model_name):
                self.models[model_name].eval()
                # print(model_name, 'eval')
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None, use_single=None):
        assert 'target_total' in batch
        assert 'target_diffuse' in batch
        assert 'target_specular' in batch
        assert 'kpcn_diffuse_in' in batch
        assert 'kpcn_specular_in' in batch
        assert 'kpcn_diffuse_buffer' in batch
        assert 'kpcn_specular_buffer' in batch
        assert 'kpcn_albedo' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None
        if self.use_llpm_buf:
            if not self.fix:
                self.models['backbone_diffuse'].zero_grad()
                self.models['backbone_specular'].zero_grad()
                p_buffers = self._manifold_forward(batch)
            else:
                with torch.no_grad():
                    p_buffers = self._manifold_forward(batch)
            # print('grad p_buffers', p_buffers['diffuse'].requires_grad)
            # if self.iters % 1000 == 1:
            #     pimg = np.mean(np.transpose(p_buffers['diffuse'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s_diffuse.png'%(self.args.model_name), pimg)

            #     pimg = np.mean(np.transpose(p_buffers['specular'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s_specular.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm11r11':
                out_manif = p_buffers
            elif self.disentanglement_option == 'm10r01':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm11r01':
                out_manif = p_buffers
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm10r11':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            batch_G = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': batch['kpcn_diffuse_in'][:, :-1],
                    'kpcn_specular_in': batch['kpcn_specular_in'][:, :-1],
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
            }
            batch_P = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'][:, :10], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'][:, :10], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
            }
        else:
            assert('should use llpm')

                # del p_buffer

        # pass denoiser
        if not self.fix:
            self.models['dncnn_G'].zero_grad()
            self.models['dncnn_P'].zero_grad()
            out = self._regress_forward(batch_G, batch_P)
        else:
            with torch.no_grad():
                out = self._regress_forward(batch_G, batch_P)

        # print('grad dncnn', out['radiance_G'].requires_grad)
        

        # make new batch for error estimation
        if self.ensemble_branches:
            self.models['interpolate_diffuse'].zero_grad()
            self.models['interpolate_specular'].zero_grad()
            if self.feature:
                # print('use_feature')
                batch_interp = {
                    'diffuse': torch.cat([out['diffuse_G'].clone().detach(), 
                                        out['diffuse_P'].clone().detach(),
                                        # crop_like(batch_G['kpcn_diffuse_in'][:, 10:], out['diffuse_G']),
                                        crop_like(batch_P['kpcn_diffuse_in'][:, 10:], out['diffuse_P']),
                                        ], dim=1),
                    'specular': torch.cat([out['specular_G'].clone().detach(), 
                                        out['specular_P'].clone().detach(),
                                        # crop_like(batch_G['kpcn_specular_in'][:, 10:], out['specular_G']),
                                        crop_like(batch_P['kpcn_specular_in'][:, 10:], out['specular_P']),
                                        ], dim=1)
                }
            else:
                # print('no_feature')
                batch_interp = {
                    'diffuse': torch.cat([out['diffuse_G'].clone().detach(), 
                                        out['diffuse_P'].clone().detach(),
                                        ], dim=1),
                    'specular': torch.cat([out['specular_G'].clone().detach(), 
                                        out['specular_P'].clone().detach(),
                                        ], dim=1)
                }
        else:
            self.models['interpolate'].zero_grad()
            if self.feature:
                # print('use_feature')
                batch_interp = {
                    'radiance': torch.cat([torch.log(out['radiance_G'].clone().detach() + 1.0), 
                                        torch.log(out['radiance_P'].clone().detach() + 1.0),
                                        crop_like(batch_G['kpcn_diffuse_in'][:, 10:], out['radiance_G']),
                                        crop_like(batch_P['kpcn_diffuse_in'][:, 10:], out['radiance_P']),
                                        ], dim=1),
                }
            else:
                # print('no_feature')
                batch_interp = {
                    'radiance': torch.cat([
                                        torch.log(out['radiance_G'].clone().detach() + 1.0), 
                                        torch.log(out['radiance_P'].clone().detach() + 1.0)
                                        ], dim=1),
                }
        # if self.feature:
        #     # print('use_feature')
        #     batch_interp = {
        #         'diffuse': torch.cat([out['diffuse_G'].clone().detach(), 
        #                             out['diffuse_P'].clone().detach(),
        #                             crop_like(batch_G['kpcn_diffuse_in'][:, 10:], out['diffuse_G']),
        #                             crop_like(batch_P['kpcn_diffuse_in'][:, 10:], out['diffuse_P']),
        #                             ], dim=1),
        #         'specular': torch.cat([out['specular_G'].clone().detach(), 
        #                             out['specular_P'].clone().detach(),
        #                             crop_like(batch_G['kpcn_specular_in'][:, 10:], out['specular_G']),
        #                             crop_like(batch_P['kpcn_specular_in'][:, 10:], out['specular_P']),
        #                             ], dim=1)
        #     }
        # else:
        #     # print('no_feature')
        #     batch_interp = {
        #         'diffuse': torch.cat([out['diffuse_G'].clone().detach(), 
        #                             out['diffuse_P'].clone().detach(),
        #                             ], dim=1),
        #         'specular': torch.cat([out['specular_G'].clone().detach(), 
        #                             out['specular_P'].clone().detach(),
        #                             ], dim=1)
        #     }
        
        interpolation_W = self._interpolation_forward(batch_interp)

        # apply interpolation
        if self.ensemble_branches:
            # print('ensemble_branches!!')
            if interpolation_W['diffuse'].shape[1] > 1:
                # print(interpolation_W['diffuse'][:, :1])
                final_diffuse = out['diffuse_G'] * interpolation_W['diffuse'][:, :1] + out['diffuse_P'] * interpolation_W['diffuse'][:, 1:]
                final_specular = out['specular_G'] * interpolation_W['specular'][:, :1] + out['specular_P'] * interpolation_W['specular'][:, 1:]
            else:
                final_diffuse = out['diffuse_G'] * interpolation_W['diffuse'] + out['diffuse_P'] * (1.0 -interpolation_W['diffuse'])
                final_specular = out['specular_G'] * interpolation_W['specular'] + out['specular_P'] * (1.0 - interpolation_W['specular'])
            final_radiance = final_diffuse * crop_like(batch['kpcn_albedo'], final_diffuse) + torch.exp(final_specular) - 1.0
            out['diffuse_I'] = final_diffuse
            out['specular_I'] = final_specular
            out['radiance_I'] = final_radiance
        else:
            if interpolation_W['radiance'].shape[1] > 1:
                final_radiance = out['radiance_G'] * interpolation_W['radiance'][:, :1] + out['radiance_P'] * interpolation_W['radiance'][:, 1:]
            else:
                final_radiance = out['radiance_G'] * interpolation_W['radiance'] + out['radiance_P'] * (1.0 -interpolation_W['radiance'])
            out['diffuse_I'] = None
            out['specular_I'] = None
            out['radiance_I'] = final_radiance
        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)
        # del loss_dict
        self._optimization()

    def _manifold_forward(self, batch):
        p_buffer_diffuse = self.models['backbone_diffuse'](batch)
        p_buffer_specular = self.models['backbone_specular'](batch)
        p_buffers = {
            'diffuse': p_buffer_diffuse,
            'specular': p_buffer_specular
        }
        return p_buffers

    def _regress_forward(self, batch_G, batch_P):
        out_G = self.models['dncnn_G'](batch_G)
        out_P = self.models['dncnn_P'](batch_P)
        out = {
            'radiance_G' : out_G['radiance'],
            'diffuse_G': out_G['diffuse'],
            'specular_G': out_G['specular'],
            'radiance_P': out_P['radiance'],
            'diffuse_P': out_P['diffuse'],
            'specular_P': out_P['specular']
        }
        return out

    def _interpolation_forward(self, batch):
        if self.ensemble_branches:
            interp_W_diffuse = self.models['interpolate_diffuse'](batch['diffuse'])
            interp_W_specular = self.models['interpolate_specular'](batch['specular'])
            interp_W = {
                'diffuse': interp_W_diffuse,
                'specular': interp_W_specular
            }
        else:
            interp_W_radiance = self.models['interpolate'](batch['radiance'])
            interp_W = {
                'radiance': interp_W_radiance,
            }
        return interp_W

    def _backward(self, batch, out, p_buffers):
        assert 'radiance_G' in out
        assert 'diffuse_G' in out
        assert 'specular_G' in out
        assert 'radiance_P' in out
        assert 'diffuse_P' in out
        assert 'specular_P' in out
        assert 'radiance_I' in out
        assert 'diffuse_I' in out
        assert 'specular_I' in out
        total_G, diffuse_G, specular_G = out['radiance_G'], out['diffuse_G'], out['specular_G']
        total_P, diffuse_P, specular_P = out['radiance_P'], out['diffuse_P'], out['specular_P']
        tgt_diffuse = crop_like(batch['target_diffuse'], diffuse_G)
        tgt_specular = crop_like(batch['target_specular'], specular_G)
        
        loss_dict = {}

        if self.train_branches: # training diffuse and specular branches

            total_I, diffuse_I, specular_I = out['radiance_I'], out['diffuse_I'], out['specular_I']
            with torch.no_grad():
                L_diffuse_G = self.loss_funcs['l_diffuse'](diffuse_G, tgt_diffuse)
                L_diffuse_P = self.loss_funcs['l_diffuse'](diffuse_P, tgt_diffuse)
                L_specular_G = self.loss_funcs['l_specular'](specular_G, tgt_specular)
                L_specular_P = self.loss_funcs['l_specular'](specular_P, tgt_specular)
            L_diffuse_I = self.loss_funcs['l_diffuse'](diffuse_I, tgt_diffuse)
            L_specular_I = self.loss_funcs['l_specular'](specular_I, tgt_specular)
            loss_dict['l_diffuse_I'] = L_diffuse_I.detach()
            loss_dict['l_specular_I'] = L_specular_I.detach()
            # L_diffuse = L_diffuse_I
            # L_specular = L_specular_I
 
            loss_dict['l_diffuse_G'] = L_diffuse_G.detach()
            loss_dict['l_diffuse_P'] = L_diffuse_P.detach()
            
            loss_dict['l_specular_G'] = L_specular_G.detach()
            loss_dict['l_specular_P'] = L_specular_P.detach()
            
            
            # TODO Write code of schedule learning & etc...
            if self.fix: # if denoisers are fixed or finetuned
                L_diffuse = L_diffuse_I
                L_specular = L_specular_I
            else: # weight 0 if not using / weight can be scheduled
                # L_diffuse = L_diffuse_I + (L_diffuse_G + L_diffuse_P) * self.weight
                # L_specular = L_specular_I + (L_specular_G + L_specular_P) * self.weight
                L_diffuse = L_diffuse_I
                L_specular = L_specular_I
            # L_diffuse = L_diffuse_G + L_diffuse_P + L_diffuse_I
            # L_specular = L_specular_G + L_specular_P + L_specular_I
            # if self.weight > 0:
            #     L_diffuse = L_diffuse_I + (L_diffuse_G + L_diffuse_P) * self.weight
            #     L_specular = L_specular_I + (L_specular_G + L_specular_P) * self.weight

            if self.manif_learn:
                p_buffer_diffuse = crop_like(p_buffers['diffuse'], diffuse_P)
                L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse)
                L_diffuse += L_manif_diffuse * self.w_manif

                p_buffer_specular = crop_like(p_buffers['specular'], specular_P)
                L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular)
                L_specular += L_manif_specular * self.w_manif

                loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach()
                loss_dict['l_manif_specular'] = L_manif_specular.detach()

                L_diffuse += L_manif_diffuse * self.w_manif
                L_specular += L_manif_specular * self.w_manif

            L_diffuse.backward()
            L_specular.backward()

        else: # single training
            total_I = out['radiance_I']
            tgt_total = crop_like(batch['target_total'], total_I)
            L_total = self.loss_funcs['l_recon'](total_I, tgt_total)
            loss_dict['l_total'] = L_total.detach()

            if self.manif_learn:
                p_buffer_diffuse = crop_like(p_buffers['diffuse'], diffuse_P)
                L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse)

                p_buffer_specular = crop_like(p_buffers['specular'], specular_P)
                L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular)

                loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach()
                loss_dict['l_manif_specular'] = L_manif_specular.detach()

                L_total += (L_manif_diffuse + L_manif_specular) * self.w_manif

            L_total.backward()

        with torch.no_grad():
            tgt_total = crop_like(batch['target_total'], total_I)
            loss_dict['rmse'] = self.loss_funcs['l_test'](total_I, tgt_total).detach()
        self.cnt += 1

    
        
        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        for model_name in self.models:
            nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            if self.fix and ('dncnn' in model_name or 'backbone' in model_name): continue 
            # print('optimiziation', model_name)
            self.optims['optim_' + model_name].step()

    def to_eval_mode(self):
        for model_name in self.models:
            self.models[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)
        # if self.error:
        #     self.m_losses['m_val_err_diff'] = torch.tensor(0.0)
        #     self.m_losses['m_val_err_spec'] = torch.tensor(0.0)

    def validate_batch(self, batch, mode=None):
        p_buffers = None

        if self.use_llpm_buf:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()
            p_buffers = self._manifold_forward(batch)

            # if self.iters % 1000 == 1:
            #     pimg = np.mean(np.transpose(p_buffers['diffuse'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s_diffuse.png'%(self.args.model_name), pimg)

            #     pimg = np.mean(np.transpose(p_buffers['specular'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
            #     pimg = np.clip(pimg, 0.0, 1.0)
            #     plt.imsave('../LLPM_results/pbuf_%s_specular.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm11r11':
                out_manif = p_buffers
            elif self.disentanglement_option == 'm10r01':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm11r01':
                out_manif = p_buffers
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm10r11':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            batch_G = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': batch['kpcn_diffuse_in'][:, :-1],
                    'kpcn_specular_in': batch['kpcn_specular_in'][:, :-1],
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
            }
            batch_P = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'][:, :10], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                    'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'][:, :10], p_buffers['specular'].mean(1), p_var_specular], 1),
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
            }
        else:
            assert('should use llpm')
        
        # pass denoiser
        out = self._regress_forward(batch_G, batch_P)

        # make new batch for interpolation
        if self.train_branches:
            if self.feature:
                # print('use_feature')
                batch_interp = {
                    'diffuse': torch.cat([out['diffuse_G'].clone().detach(), 
                                        out['diffuse_P'].clone().detach(),
                                        # crop_like(batch_G['kpcn_diffuse_in'][:, 10:], out['diffuse_G']),
                                        crop_like(batch_P['kpcn_diffuse_in'][:, 10:], out['diffuse_P']),
                                        ], dim=1),
                    'specular': torch.cat([out['specular_G'].clone().detach(), 
                                        out['specular_P'].clone().detach(),
                                        # crop_like(batch_G['kpcn_specular_in'][:, 10:], out['specular_G']),
                                        crop_like(batch_P['kpcn_specular_in'][:, 10:], out['specular_P']),
                                        ], dim=1)
                }
            else:
                # print('no_feature')
                batch_interp = {
                    'diffuse': torch.cat([out['diffuse_G'].clone().detach(), 
                                        out['diffuse_P'].clone().detach(),
                                        ], dim=1),
                    'specular': torch.cat([out['specular_G'].clone().detach(), 
                                        out['specular_P'].clone().detach(),
                                        ], dim=1)
                }
        else:
            if self.feature:
                # print('use_feature')
                batch_interp = {
                    'radiance': torch.cat([out['radiance_G'].clone().detach(), 
                                        out['radiance_P'].clone().detach(),
                                        crop_like(batch_G['kpcn_diffuse_in'][:, 10:], out['radiance_G']),
                                        crop_like(batch_P['kpcn_diffuse_in'][:, 10:], out['radiance_P']),
                                        ], dim=1),
                }
            else:
                # print('no_feature')
                batch_interp = {
                    'radiance': torch.cat([out['radiance_G'].clone().detach(),
                                        out['radiance_P'].clone().detach(),
                                        ], dim=1),
                }

        interpolation_W = self._interpolation_forward(batch_interp)

        # apply interpolation
        if self.train_branches:
            if interpolation_W['diffuse'].shape[1] > 1:
                final_diffuse = out['diffuse_G'] * interpolation_W['diffuse'][:, :1] + out['diffuse_P'] * interpolation_W['diffuse'][:, 1:]
                final_specular = out['specular_G'] * interpolation_W['specular'][:, :1] + out['specular_P'] * interpolation_W['specular'][:, 1:]
            else:
                final_diffuse = out['diffuse_G'] * interpolation_W['diffuse'] + out['diffuse_P'] * (1.0 -interpolation_W['diffuse'])
                final_specular = out['specular_G'] * interpolation_W['specular'] + out['specular_P'] * (1.0 - interpolation_W['specular'])
            final_radiance = final_diffuse * crop_like(batch['kpcn_albedo'], final_diffuse) + torch.exp(final_specular) - 1.0
            out['diffuse_I'] = final_diffuse
            out['specular_I'] = final_specular
            out['radiance_I'] = final_radiance
        else:
            if interpolation_W['radiance'].shape[1] > 1:
                final_radiance = out['radiance_G'] * interpolation_W['radiance'][:, :1] + out['radiance_P'] * interpolation_W['radiance'][:, 1:]
            else:
                final_radiance = out['radiance_G'] * interpolation_W['radiance'] + out['radiance_P'] * (1.0 -interpolation_W['radiance'])
            final_diffuse, final_specular = None, None

        rad_dict = {
            'g_radiance': out['radiance_G'], 'g_diffuse': out['diffuse_G'], 'g_specular': out['specular_G'],
            'p_radiance': out['radiance_P'], 'p_diffuse': out['diffuse_P'], 'p_specular': out['specular_P'],
            'radiance': final_radiance, 'diffuse': final_diffuse, 'specular': final_specular
            }
        kernel_dict = {'diffuse': None,
                      'specular': None}
        
        tgt_total = crop_like(batch['target_total'], final_radiance)

        L_total = self.loss_funcs['l_test'](final_radiance, tgt_total)

        score_dict = {}
        if self.train_branches:
            if interpolation_W['diffuse'].shape[1] > 1:
                score_dict['weight_diffuse_G'] = interpolation_W['diffuse'][:, :1]
                score_dict['weight_diffuse_P'] = interpolation_W['diffuse'][:, 1:]
                score_dict['weight_specular_G'] = interpolation_W['specular'][:, :1]
                score_dict['weight_specular_P'] = interpolation_W['specular'][:, 1:]
        else:
            if interpolation_W['radiance'].shape[1] > 1:
                score_dict['weight_diffuse_G'] = interpolation_W['radiance'][:, :1]
                score_dict['weight_diffuse_P'] = interpolation_W['radiance'][:, 1:]

        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return final_radiance,  p_buffers, rad_dict, kernel_dict, score_dict

    def get_epoch_summary(self, mode, norm):
        if mode == 'train':
            losses = {}
            print('[][][]', end=' ')
            for key in self.m_losses:
                if 'm_val' in key:
                    continue
                tr_l_tmp = self.m_losses[key] / (norm * 2)
                tr_l_tmp *= 1000
                print('%s: %.3fE-3'%(key, tr_l_tmp), end='\t')
                losses[key] = tr_l_tmp
                self.m_losses[key] = torch.tensor(0.0, device=self.m_losses[key].device)
            print('')
            return losses #-1.0
        else:
            # if not self.error:
            losses = {}
            for key in self.m_losses:
                if 'm_val' not in key:
                    continue
                losses[key] = self.m_losses[key].item() / (norm * 2)
            return losses