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
        
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

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

    def validate_batch(self, batch, vis=None):
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
        
        out = self._regress_forward(batch)
        rad_dict = {'diffuse': out['diffuse'],
                    'specular': out['specular']}
        kernel_dict = {'diffuse':out['k_diffuse'],
                      'specular': out['k_specular']}
        
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


class AdvKPCNInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_adv=False, w_adv=0.0001):
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
        
        super(AdvKPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.use_adv = use_adv
        self.cnt = 0
        self.epoch = 0
        self.w_adv = w_adv

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
        
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        self._optimization()

    def _manifold_forward(self, batch):
        p_buffer_diffuse = self.models['backbone_diffuse'](batch)
        p_buffer_specular = self.models['backbone_specular'](batch)
        p_buffers = {
            'diffuse': p_buffer_diffuse,
            'specular': p_buffer_specular
        }
        return p_buffers

    def _regress_forward(self, batch, vis=False):
        # print(next(self.models['dncnn'].parameters()).device)
        return self.models['dncnn'](batch, vis)

    def _backward(self, batch, out, p_buffers):
        assert 'radiance' in out
        assert 'diffuse' in out
        assert 'specular' in out

        total, diffuse, specular = out['radiance'], out['diffuse'], out['specular']
        g_total, g_diffuse, g_specular = out['g_radiance'], out['g_diffuse'], out['g_specular']
        if self.use_adv:
            s_f_diffuse, s_f_specular = out['s_f_diffuse'], out['s_f_specular']
            s_r_diffuse, s_r_specular = out['s_r_diffuse'], out['s_r_specular']
        loss_dict = {}
        tgt_total = crop_like(batch['target_total'], total)
        if self.train_branches: # training diffuse and specular branches
            tgt_diffuse = crop_like(batch['target_diffuse'], diffuse)
            L_diffuse = self.loss_funcs['l_diffuse'](diffuse, tgt_diffuse)
            loss_dict['l_diffuse'] = L_diffuse.detach()
            L_g_diffuse = self.loss_funcs['l_diffuse'](g_diffuse, tgt_diffuse)
            loss_dict['l_g_diffuse'] = L_g_diffuse.detach()
            L_diffuse += L_g_diffuse
            if self.use_adv:
                L_adv_f_diffuse = self.loss_funcs['l_adv'](s_f_diffuse, torch.zeros_like(s_f_diffuse))
                L_adv_r_diffuse = self.loss_funcs['l_adv'](s_r_diffuse, torch.ones_like(s_r_diffuse))
                L_adv_diffuse = (L_adv_f_diffuse + L_adv_r_diffuse) / 2
                loss_dict['l_adv_diffuse'] = L_adv_diffuse.detach()
                L_diffuse += L_adv_diffuse * self.w_adv
            loss_dict['l_total_diffuse'] = L_diffuse.detach()

            tgt_specular = crop_like(batch['target_specular'], specular)
            L_specular = self.loss_funcs['l_specular'](specular, tgt_specular)
            loss_dict['l_specular'] = L_specular.detach()
            L_g_specular = self.loss_funcs['l_specular'](g_specular, tgt_specular)
            loss_dict['l_g_specular'] = L_g_specular.detach()
            L_specular += L_g_specular
            if self.use_adv:
                L_adv_f_specular = self.loss_funcs['l_adv'](s_f_specular, torch.zeros_like(s_f_specular))
                L_adv_r_specular = self.loss_funcs['l_adv'](s_r_specular, torch.ones_like(s_r_specular))
                L_adv_specular = (L_adv_f_specular + L_adv_r_specular) / 2
                loss_dict['l_adv_specular'] = L_adv_specular.detach()
                L_specular += L_adv_specular * self.w_adv
            loss_dict['l_total_specular'] = L_specular.detach()
    

            # joint training refinement & denoising
            
            if self.manif_learn:
                p_buffer_diffuse = crop_like(p_buffers['diffuse'], diffuse)
                L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse)
                L_diffuse += L_manif_diffuse * self.w_manif

                p_buffer_specular = crop_like(p_buffers['specular'], specular)
                L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular)
                L_specular += L_manif_specular * self.w_manif

                loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach()
                loss_dict['l_manif_specular'] = L_manif_specular.detach()
            
            L_diffuse.backward()
            L_specular.backward()
            # L_g_diffuse.backward()
            # L_g_specular.backward()

            with torch.no_grad():
                L_total = self.loss_funcs['l_recon'](total, tgt_total)
                loss_dict['l_total'] = L_total.detach()
                L_g_total = self.loss_funcs['l_recon'](g_total, tgt_total)
                loss_dict['l_g_total'] = L_total.detach()
        else: # post-training the entire system
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            loss_dict['l_total'] = L_total.detach()
            L_total.backward()
        
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

    def validate_batch(self, batch, vis=False):
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
        
        out = self._regress_forward(batch, vis)
        rad_dict = {'diffuse': out['diffuse'],
                    'specular': out['specular']}
        kernel_dict = {'diffuse':out['k_diffuse'],
                      'specular': out['k_specular']}
        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        rad_dict['g_radiance'] = out['g_radiance']
        rad_dict['g_diffuse'] = out['g_diffuse']
        rad_dict['g_specular'] = out['g_specular']
        kernel_dict['g_diffuse'] = out['k_g_diffuse']
        kernel_dict['g_specular'] = out['k_g_specular']
        score_dict = {
            'fake_diffuse': out['s_f_diffuse'],
            'fake_specular': out['s_f_specular'],
            'real_diffuse': out['s_r_diffuse'],
            'real_specular': out['s_r_specular']
        }

        L_gbuf_total = self.loss_funcs['l_test'](out['g_radiance'], tgt_total)
        if self.m_losses['m_gbuf_val'] == 0.0 and self.m_losses['m_gbuf_val'].device != L_gbuf_total.device:
            self.m_losses['m_gbuf_val'] = torch.tensor(0.0, device=L_gbuf_total.device)
        L_gbuf_total = self.loss_funcs['l_test'](out['g_radiance'], tgt_total)
        self.m_losses['m_gbuf_val'] += L_gbuf_total.detach()
        return out['radiance'],  p_buffers, rad_dict, kernel_dict, score_dict

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


class NewAdvKPCNInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_adv=False, w_adv=0.0001):
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
        
        super(NewAdvKPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.use_adv = use_adv
        self.cnt = 0
        self.epoch = 0
        self.w_adv = w_adv
        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
        self.separate = args.separate
        self.manif_learn = manif_learn

    def __str__(self):
        return 'KPCNInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None, use_single=False):
        if not use_single:
            assert 'target_total' in batch
            assert 'target_diffuse' in batch
            assert 'target_specular' in batch
            assert 'kpcn_diffuse_in' in batch
            assert 'kpcn_specular_in' in batch
            assert 'kpcn_diffuse_buffer' in batch
            assert 'kpcn_specular_buffer' in batch
            assert 'kpcn_albedo' in batch
        else:
            assert 'target_total' in batch
            assert 'kpcn_buffer' in batch
            assert 'kpcn_in' in batch
            assert 'kpcn_albedo' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None
        if not self.separate:
            if self.manif_learn:
                self.models['backbone'].zero_grad()
                pbuffer = self._manifold_forward(batch)
                pbuffer_var = pbuffer.var(1).mean(1, keepdims=True).detach()
                pbuffer_var /= pbuffer.shape[1]
                out_manif = torch.cat([pbuffer.mean(1), pbuffer_var])
                batch['paths'] = out_manif
            self.models['dncnn'].zero_grad()
            out = self._regress_forward(batch, separate=self.separate)

            loss_dict = self._backward(batch, out, out_manif)

            if grad_hook_mode: # do not update this model
                return

            self._logging(loss_dict)

            self._optimization()

        else:
            loss_dict = self._train_gan(batch, False)


    def _train_gan(self, batch, vis=False):
        loss_dict = {}
        real_target = torch.tensor([1.0]).cuda()
        fake_target = torch.tensor([0.0]).cuda()
        # train discriminator first
        self.models['dis'].zero_grad()
        out = self.models['dncnn'](batch, False)
        g_rad, p_rad= out['g_radiance'], out['p_radiance']
        dis_batch = {
        'target_in': torch.cat([crop_like(batch['target_total'], g_rad), crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'g_rad_in': torch.cat([g_rad, crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'p_rad_in': torch.cat([p_rad, crop_like(batch["kpcn_in"][:,20:], p_rad), crop_like(batch["paths"].mean(1), p_rad)], 1),
        }
        dis_out = self.models['dis'](dis_batch, mode='adv_2')
        g_score, p_score, gt_score = dis_out['fake'], dis_out['fake_2'], dis_out['real']
        L_D_g = F.binary_cross_entropy_with_logits(g_score, fake_target.expand_as(g_score))
        L_D_p = F.binary_cross_entropy_with_logits(p_score, fake_target.expand_as(p_score))
        L_D_gt = F.binary_cross_entropy_with_logits(gt_score, real_target.expand_as(gt_score))
        loss_dict['l_D_g'] = L_D_g.detach()
        loss_dict['l_D_p'] = L_D_p.detach()
        loss_dict['l_D_gt'] = L_D_gt.detach()
        L_D = 0.25 * (L_D_g + L_D_p + 2.0 * L_D_gt)
        loss_dict['l_D'] = L_D_gt.detach()
        L_D.backward()
        self.optims['optim_dis'].step()
        del L_D
        del g_score, p_score, gt_score

        # train denoiser(generator) later
        self.models['dncnn'].zero_grad()
        out = self.models['dncnn'](batch, True)
        g_rad, p_rad, g_kernel, p_kernel = out['g_radiance'], out['p_radiance'], out['g_kernel'], out['p_kernel']
        dis_batch = {
        'target_in': torch.cat([batch['target_total'], crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'g_rad_in': torch.cat([g_rad, crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'p_rad_in': torch.cat([p_rad, crop_like(batch["kpcn_in"][:,20:], p_rad), crop_like(batch["paths"].mean(1), p_rad)], 1),
        }
        dis_out = self.models['dis'](dis_batch, mode='adv_2')
        g_score, p_score, gt_score = dis_out['fake'], dis_out['fake_2'], dis_out['real']
        new_kernel = (g_kernel * g_score + p_kernel * p_score).contiguous()
        buffer = crop_like(batch["kpcn_buffer"], new_kernel).contiguous()
        final_radiance, _ = self.kernel_apply(buffer, new_kernel)

        tgt_total = crop_like(batch['target_total'], final_radiance)
        L_g = self.loss_funcs['l_recon'](tgt_total, g_rad)
        L_p = self.loss_funcs['l_recon'](tgt_total, p_rad)
        L_final = self.loss_funcs['l_recon'](tgt_total, final_radiance)
        L_recon = 0.25 * (L_g + L_p  + 2.0 * L_final)
        loss_dict['l_g'] = L_g.detach()
        loss_dict['l_p'] = L_p.detach()
        loss_dict['l_final'] = L_p.detach()
        # adversarial losses
        L_g_adv = self.loss_funcs['l_adv'](g_score, torch.ones_like(g_score).cuda())
        L_p_adv = self.loss_funcs['l_adv'](p_score, torch.ones_like(p_score).cuda())
        dis_batch = {
            'final_in': torch.cat([final_radiance, crop_like(batch["kpcn_in"][:,20:], final_radiance), crop_like(batch["paths"].mean(1), final_radiance)], 1),
        }
        final_score = self.models['dis'](dis_batch, mode='final')['fake']
        L_final_adv = self.loss_funcs['l_adv'](final_score, torch.ones_like(final_score).cuda())
        loss_dict['l_g_adv'] = L_g_adv.detach()
        loss_dict['l_p_adv'] = L_p_adv.detach()
        loss_dict['l_final_adv'] = L_final_adv.detach()
        L_adv = 0.25 * (L_g_adv + L_p_adv + 2.0 * L_final_adv)

        L_G = L_recon + self.w_adv * L_adv
        L_G.backward()
        self.optims['optim_dncnn'].step()

        # logging
        self._logging(loss_dict)


    def _manifold_forward(self, batch):
        p_buffer = self.models['backbone'](batch)
        return p_buffer

    def _regress_forward(self, batch, vis=False, separate=False):
        # print(next(self.models['dncnn'].parameters()).device)
        # print('_regress_forward', separate)
        if not separate:
            return self.models['dncnn'](batch, vis)
        else:
            out = self.models['dncnn'](batch, True)
            g_rad, p_rad, g_kernel, p_kernel = out['g_radiance'], out['p_radiance'], out['g_kernel'], out['p_kernel']
            dis_batch = {
            'target_in': torch.cat([batch['target_total'], crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
            'g_rad_in': torch.cat([g_rad, crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
            'p_rad_in': torch.cat([p_rad, crop_like(batch["kpcn_in"][:,20:], p_rad), crop_like(batch["paths"].mean(1), p_rad)], 1),
            }
            self.models['dis'].zero_grad()
            dis_out = self.models['dis'](dis_batch, mode='adv_2')
            g_score, p_score, gt_score = dis_out['fake'], dis_out['fake_2'], dis_out['real']
            new_kernel = (g_kernel * g_score + p_kernel * p_score).contiguous()

            buffer = crop_like(batch["kpcn_buffer"], new_kernel).contiguous()
            final_radiance, _ = self.kernel_apply(buffer, new_kernel)
            return dict(
                radiance=final_radiance, g_radiance=g_rad, p_radiance=p_rad,
                g_score=g_score, p_score=p_score, gt_score=gt_score,
                g_kernel=g_kernel, p_kernel=p_kernel, new_kernel=new_kernel
            )


    def _backward(self, batch, out, p_buffers):
        assert 'radiance' in out
        # assert 'diffuse' in out
        # assert 'specular' in out
        if self.manif_learn:
            p_buffer = crop_like

        final_radiance, g_radiance, p_radiance = out['radiance'], out['g_radiance'], out['p_radiance']
        g_score, p_score, gt_score = out['g_score'], out['p_score'], out['gt_score']
        loss_dict = {}
        tgt_total = crop_like(batch['target_total'], final_radiance)
        # Loss from gbuf_only branch
        L_g = self.loss_funcs['l_recon'](tgt_total, g_radiance)
        loss_dict['l_g'] = L_g.detach()
        # Loss from pbuf_only branch
        L_p = self.loss_funcs['l_recon'](tgt_total, p_radiance)
        loss_dict['l_p'] = L_p.detach()
        # Loss for adversarial trainng
        L_g_adv = self.loss_funcs['l_adv'](g_score, torch.zeros_like(g_score))
        L_p_adv = self.loss_funcs['l_adv'](p_score, torch.zeros_like(p_score))
        L_gt_adv = self.loss_funcs['l_adv'](gt_score, torch.ones_like(gt_score))
        loss_dict['l_g_adv'] = L_g_adv.detach()
        loss_dict['l_p_adv'] = L_p_adv.detach()
        loss_dict['l_gt_adv'] = L_gt_adv.detach()
        L_adv = L_g_adv + L_p_adv + L_gt_adv * 2.0 # for balancing effect of noisy&gt image
        # Loss for final denoising
        L_final = self.loss_funcs['l_recon'](tgt_total, final_radiance)
        loss_dict['l_final'] = L_final.detach()
        L_total = (L_g + L_p + 2.0 * L_final) + L_adv * self.w_adv
        # L_total = L_final + L_adv * self.w_adv
        loss_dict['l_total'] = L_total.detach()
        L_total.backward()


        with torch.no_grad():
            L_total = self.loss_funcs['l_recon'](tgt_total, final_radiance)
            loss_dict['l_total'] = L_total.detach()
        # L_total = self.loss_funcs['l_recon'](total, tgt_total)
        # loss_dict['l_total'] = L_total.detach()
        # L_total.backward()
        
        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](final_radiance, tgt_total).detach()
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

    def validate_batch(self, batch, vis=False):
        p_buffers = None

        out = self._regress_forward(batch, vis, self.separate)
        rad_dict = {
                    'radiance': out['radiance'],
                    'g_radiance': out['g_radiance'],
                    'p_radiance': out['p_radiance'],
                    }
        kernel_dict = {
                        'g_kernel':out['g_kernel'],
                        'p_kernel': out['p_kernel'],
                        'new_kernel': out['new_kernel']
                    }
        score_dict = {
            'g_score': out['g_score'],
            'p_score': out['p_score'],
            'gt_score': out['gt_score'],
        }

        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out['radiance'],  p_buffers, rad_dict, kernel_dict, score_dict

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
            return losses # -1.0
        else:
            return self.m_losses['m_val'].item() / (norm * 2)


class NewAdvKPCNInterface1(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_adv=False, w_adv=0.0001):
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
        
        super(NewAdvKPCNInterface1, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.use_adv = use_adv
        self.cnt = 0
        self.epoch = 0
        self.w_adv = w_adv
        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
        self.separate = args.separate
        self.manif_learn = manif_learn

    def __str__(self):
        return 'KPCNInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None, use_single=False):
        if not use_single:
            assert 'target_total' in batch
            assert 'target_diffuse' in batch
            assert 'target_specular' in batch
            assert 'kpcn_diffuse_in' in batch
            assert 'kpcn_specular_in' in batch
            assert 'kpcn_diffuse_buffer' in batch
            assert 'kpcn_specular_buffer' in batch
            assert 'kpcn_albedo' in batch
        else:
            assert 'target_total' in batch
            assert 'kpcn_buffer' in batch
            assert 'kpcn_in' in batch
            assert 'kpcn_albedo' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None
        
        if not self.separate:
            if self.manif_learn:
                self.models['backbone_diffuse'].zero_grad()
                self.models['backbone_specular'].zero_grad()
                p_buffers = self._manifold_forward(batch)

                _, _, c, _, _ = p_buffers['diffuse'].shape
                assert c >= 2
                if self.disentanglement_option == 'm11r11':
                    out_manif = p_buffers
                # elif self.disentanglement_option == 'm10r01':
                #     out_manif = {
                #             'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                #             'specular': p_buffers['specular'][:,:,c//2:,...]
                #     }
                #     p_buffers = {
                #             'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                #             'specular': p_buffers['specular'][:,:,:c//2,...]
                #     }
                # elif self.disentanglement_option == 'm11r01':
                #     out_manif = p_buffers
                #     p_buffers = {
                #             'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                #             'specular': p_buffers['specular'][:,:,:c//2,...]
                #     }
                # elif self.disentanglement_option == 'm10r11':
                #     out_manif = {
                #             'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                #             'specular': p_buffers['specular'][:,:,c//2:,...]
                #     }
                else:
                    assert NotImplementedError('use disentangle option m11r11')

                p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
                p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
                p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
                p_var_specular /= p_buffers['specular'].shape[1]

                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': batch['kpcn_diffuse_in'],
                    'kpcn_specular_in': batch['kpcn_specular_in'],
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                    'paths_diffuse': torch.cat([p_buffers['diffuse'].mean(1), p_var_diffuse], dim=1),
                    'paths_specular': torch.cat([p_buffers['specular'].mean(1), p_var_specular], dim=1),
                }
            else:
                p_var = batch['paths'].var(1).mean(1, keepdims=True).detach()
                p_var /= batch['paths'].shape[1] # spp
                batch = {
                    'target_total': batch['target_total'],
                    'target_diffuse': batch['target_diffuse'],
                    'target_specular': batch['target_specular'],
                    'kpcn_diffuse_in': batch['kpcn_diffuse_in'],
                    'kpcn_specular_in': batch['kpcn_specular_in'],
                    'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                    'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                    'kpcn_albedo': batch['kpcn_albedo'],
                    'paths_diffuse': torch.cat([batch['paths'].mean(1), p_var], dim=1),
                    'paths_specular': torch.cat([batch['paths'].mean(1), p_var], dim=1),
                }


            self.models['dncnn'].zero_grad()
            out = self._regress_forward(batch, separate=self.separate)

            loss_dict = self._backward(batch, out, out_manif)

            if grad_hook_mode: # do not update this model
                return

            self._logging(loss_dict)

            self._optimization()

        else:
            loss_dict = self._train_gan(batch, False)


    def _train_gan(self, batch, vis=False):
        loss_dict = {}
        real_target = torch.tensor([1.0]).cuda()
        fake_target = torch.tensor([0.0]).cuda()
        # train discriminator first
        self.models['dis'].zero_grad()
        out = self.models['dncnn'](batch, False)
        g_rad, p_rad= out['g_radiance'], out['p_radiance']
        dis_batch = {
        'target_in': torch.cat([crop_like(batch['target_total'], g_rad), crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'g_rad_in': torch.cat([g_rad, crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'p_rad_in': torch.cat([p_rad, crop_like(batch["kpcn_in"][:,20:], p_rad), crop_like(batch["paths"].mean(1), p_rad)], 1),
        }
        dis_out = self.models['dis'](dis_batch, mode='adv_2')
        g_score, p_score, gt_score = dis_out['fake'], dis_out['fake_2'], dis_out['real']
        L_D_g = F.binary_cross_entropy_with_logits(g_score, fake_target.expand_as(g_score))
        L_D_p = F.binary_cross_entropy_with_logits(p_score, fake_target.expand_as(p_score))
        L_D_gt = F.binary_cross_entropy_with_logits(gt_score, real_target.expand_as(gt_score))
        loss_dict['l_D_g'] = L_D_g.detach()
        loss_dict['l_D_p'] = L_D_p.detach()
        loss_dict['l_D_gt'] = L_D_gt.detach()
        L_D = 0.25 * (L_D_g + L_D_p + 2.0 * L_D_gt)
        loss_dict['l_D'] = L_D_gt.detach()
        L_D.backward()
        self.optims['optim_dis'].step()
        del L_D
        del g_score, p_score, gt_score

        # train denoiser(generator) later
        self.models['dncnn'].zero_grad()
        out = self.models['dncnn'](batch, True)
        g_rad, p_rad, g_kernel, p_kernel = out['g_radiance'], out['p_radiance'], out['g_kernel'], out['p_kernel']
        dis_batch = {
        'target_in': torch.cat([batch['target_total'], crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'g_rad_in': torch.cat([g_rad, crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
        'p_rad_in': torch.cat([p_rad, crop_like(batch["kpcn_in"][:,20:], p_rad), crop_like(batch["paths"].mean(1), p_rad)], 1),
        }
        dis_out = self.models['dis'](dis_batch, mode='adv_2')
        g_score, p_score, gt_score = dis_out['fake'], dis_out['fake_2'], dis_out['real']
        new_kernel = (g_kernel * g_score + p_kernel * p_score).contiguous()
        buffer = crop_like(batch["kpcn_buffer"], new_kernel).contiguous()
        final_radiance, _ = self.kernel_apply(buffer, new_kernel)

        tgt_total = crop_like(batch['target_total'], final_radiance)
        L_g = self.loss_funcs['l_recon'](tgt_total, g_rad)
        L_p = self.loss_funcs['l_recon'](tgt_total, p_rad)
        L_final = self.loss_funcs['l_recon'](tgt_total, final_radiance)
        L_recon = 0.25 * (L_g + L_p  + 2.0 * L_final)
        loss_dict['l_g'] = L_g.detach()
        loss_dict['l_p'] = L_p.detach()
        loss_dict['l_final'] = L_p.detach()
        # adversarial losses
        L_g_adv = self.loss_funcs['l_adv'](g_score, torch.ones_like(g_score).cuda())
        L_p_adv = self.loss_funcs['l_adv'](p_score, torch.ones_like(p_score).cuda())
        dis_batch = {
            'final_in': torch.cat([final_radiance, crop_like(batch["kpcn_in"][:,20:], final_radiance), crop_like(batch["paths"].mean(1), final_radiance)], 1),
        }
        final_score = self.models['dis'](dis_batch, mode='final')['fake']
        L_final_adv = self.loss_funcs['l_adv'](final_score, torch.ones_like(final_score).cuda())
        loss_dict['l_g_adv'] = L_g_adv.detach()
        loss_dict['l_p_adv'] = L_p_adv.detach()
        loss_dict['l_final_adv'] = L_final_adv.detach()
        L_adv = 0.25 * (L_g_adv + L_p_adv + 2.0 * L_final_adv)

        L_G = L_recon + self.w_adv * L_adv
        L_G.backward()
        self.optims['optim_dncnn'].step()

        # logging
        self._logging(loss_dict)


    def _manifold_forward(self, batch):
        p_buffer_diffuse = self.models['backbone_diffuse'](batch)
        p_buffer_specular = self.models['backbone_specular'](batch)
        p_buffers = {
            'diffuse': p_buffer_diffuse,
            'specular': p_buffer_specular
        }
        return p_buffers
        return 

    def _regress_forward(self, batch, vis=False, separate=False):
        # print(next(self.models['dncnn'].parameters()).device)
        # print('_regress_forward', separate)
        if not separate:
            return self.models['dncnn'](batch, vis)
        else:
            out = self.models['dncnn'](batch, True)
            g_rad, p_rad, g_kernel, p_kernel = out['g_radiance'], out['p_radiance'], out['g_kernel'], out['p_kernel']
            dis_batch = {
            'target_in': torch.cat([batch['target_total'], crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
            'g_rad_in': torch.cat([g_rad, crop_like(batch["kpcn_in"][:,20:], g_rad), crop_like(batch["paths"].mean(1), g_rad)], 1),
            'p_rad_in': torch.cat([p_rad, crop_like(batch["kpcn_in"][:,20:], p_rad), crop_like(batch["paths"].mean(1), p_rad)], 1),
            }
            self.models['dis'].zero_grad()
            dis_out = self.models['dis'](dis_batch, mode='adv_2')
            g_score, p_score, gt_score = dis_out['fake'], dis_out['fake_2'], dis_out['real']
            new_kernel = (g_kernel * g_score + p_kernel * p_score).contiguous()

            buffer = crop_like(batch["kpcn_buffer"], new_kernel).contiguous()
            final_radiance, _ = self.kernel_apply(buffer, new_kernel)
            return dict(
                radiance=final_radiance, g_radiance=g_rad, p_radiance=p_rad,
                g_score=g_score, p_score=p_score, gt_score=gt_score,
                g_kernel=g_kernel, p_kernel=p_kernel, new_kernel=new_kernel
            )


    def _backward(self, batch, out, p_buffers):
        assert 'radiance' in out
        # assert 'diffuse' in out
        # assert 'specular' in out

        final_radiance, g_radiance, p_radiance = out['radiance'], out['g_radiance'], out['p_radiance']
        final_diffuse, g_diffuse, p_diffuse = out['diffuse'], out['g_diffuse'], out['p_diffuse']
        final_specular, g_specular, p_specular = out['specular'], out['g_specular'], out['p_specular']
        gt_diff_score, gt_spec_score = out['s_diffuse'], out['s_specular']
        g_diff_score, g_spec_score = out['s_g_diffuse'], out['s_g_specular']
        p_diff_score, p_spec_score = out['s_p_diffuse'], out['s_p_specular']
        loss_dict = {}
        tgt_diffuse = crop_like(batch['target_diffuse'], final_radiance)
        tgt_specular = crop_like(batch['target_specular'], final_radiance)
        tgt_total = crop_like(batch['target_total'], final_radiance)
        
        L_diff = self.loss_funcs['l_diffuse'](tgt_diffuse, final_diffuse)
        loss_dict['l_diff'] = L_diff.detach()
        L_spec = self.loss_funcs['l_specular'](tgt_specular, final_specular)
        loss_dict['l_spec'] = L_spec.detach()
        L_g_diff = self.loss_funcs['l_diffuse'](tgt_diffuse, g_diffuse)
        loss_dict['l_g_diff'] = L_g_diff.detach()
        L_g_spec = self.loss_funcs['l_specular'](tgt_specular, g_specular)
        loss_dict['l_g_spec'] = L_g_spec.detach()
        L_p_diff = self.loss_funcs['l_diffuse'](tgt_diffuse, p_diffuse)
        loss_dict['l_p_diff'] = L_p_diff.detach()
        L_p_spec = self.loss_funcs['l_specular'](tgt_specular, p_specular)
        loss_dict['l_p_spec'] = L_p_spec.detach()
        # L_recon_ = (L_diff + L_spec + L_g_diff + L_g_spec + L_p_diff + L_p_spec) / 6.0
        L_recon_diff = L_diff + L_g_diff + L_p_diff
        L_recon_spec = L_spec + L_g_spec + L_p_spec
        # L_recon_diff = L_diff
        # L_recon_spec = L_spec
        # Loss for adversarial trainng
        L_g_diff_adv = self.loss_funcs['l_adv'](g_diff_score, torch.zeros_like(g_diff_score))
        L_p_diff_adv = self.loss_funcs['l_adv'](p_diff_score, torch.zeros_like(p_diff_score))
        L_g_spec_adv = self.loss_funcs['l_adv'](g_spec_score, torch.zeros_like(g_spec_score))
        L_p_spec_adv = self.loss_funcs['l_adv'](p_spec_score, torch.zeros_like(p_spec_score))
        L_gt_diff_adv = self.loss_funcs['l_adv'](gt_diff_score, torch.ones_like(gt_diff_score))
        L_gt_spec_adv = self.loss_funcs['l_adv'](gt_spec_score, torch.ones_like(gt_spec_score))
        loss_dict['l_g_diff_adv'] = L_g_diff_adv.detach()
        loss_dict['l_p_diff_adv'] = L_p_diff_adv.detach()
        loss_dict['l_gt_diff_adv'] = L_gt_diff_adv.detach()
        loss_dict['l_g_spec_adv'] = L_g_spec_adv.detach()
        loss_dict['l_p_spec_adv'] = L_p_spec_adv.detach()
        loss_dict['l_gt_spec_adv'] = L_gt_spec_adv.detach()
        # L_adv = (L_g_diff_adv + L_p_diff_adv + L_gt_diff_adv + L_g_spec_adv + L_p_spec_adv + L_gt_spec_adv) / 6.0# for balancing effect of noisy&gt image
        L_adv_diff = L_g_diff_adv + L_p_diff_adv + L_gt_diff_adv
        L_adv_spec = L_g_spec_adv + L_p_spec_adv + L_gt_spec_adv
        # L_adv_diff = L_g_diff_adv + L_p_diff_adv + L_gt_diff_adv * 2.0 
        # L_adv_spec = L_g_spec_adv + L_p_spec_adv + L_gt_spec_adv * 2.0

        L_total_diff = L_recon_diff + L_adv_diff * self.w_adv
        L_total_spec = L_recon_spec + L_adv_spec * self.w_adv

        if self.manif_learn:
            p_buffer_diffuse = crop_like(p_buffers['diffuse'], g_diffuse)
            L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse)

            p_buffer_specular = crop_like(p_buffers['specular'], g_specular)
            L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular)

            loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach()
            loss_dict['l_manif_specular'] = L_manif_specular.detach()

            L_total_diff += L_manif_diffuse * self.w_manif
            L_total_spec += L_manif_specular * self.w_manif

        L_total_diff.backward()
        L_total_spec.backward()
        
        # Loss for final denoising
        L_final = self.loss_funcs['l_recon'](tgt_total, final_radiance)
        loss_dict['l_final'] = L_final.detach()

        with torch.no_grad():
            L_total = self.loss_funcs['l_recon'](tgt_total, final_radiance)
            loss_dict['l_total'] = L_total.detach()
        # L_total = self.loss_funcs['l_recon'](total, tgt_total)
        # loss_dict['l_total'] = L_total.detach()
        # L_total.backward()
        
        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](final_radiance, tgt_total).detach()
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

    def validate_batch(self, batch, vis=False):
        p_buffers = None
        if self.manif_learn:
            p_buffers = self._manifold_forward(batch)

            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm11r11':
                out_manif = p_buffers
            # elif self.disentanglement_option == 'm10r01':
            #     out_manif = {
            #             'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
            #             'specular': p_buffers['specular'][:,:,c//2:,...]
            #     }
            #     p_buffers = {
            #             'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
            #             'specular': p_buffers['specular'][:,:,:c//2,...]
            #     }
            # elif self.disentanglement_option == 'm11r01':
            #     out_manif = p_buffers
            #     p_buffers = {
            #             'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
            #             'specular': p_buffers['specular'][:,:,:c//2,...]
            #     }
            # elif self.disentanglement_option == 'm10r11':
            #     out_manif = {
            #             'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
            #             'specular': p_buffers['specular'][:,:,c//2:,...]
            #     }
            else:
                assert NotImplementedError('use disentangle option m11r11')

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            batch = {
                'target_total': batch['target_total'],
                'target_diffuse': batch['target_diffuse'],
                'target_specular': batch['target_specular'],
                'kpcn_diffuse_in': batch['kpcn_diffuse_in'],
                'kpcn_specular_in': batch['kpcn_specular_in'],
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                'kpcn_albedo': batch['kpcn_albedo'],
                'paths_diffuse': torch.cat([p_buffers['diffuse'].mean(1), p_var_diffuse], dim=1),
                'paths_specular': torch.cat([p_buffers['specular'].mean(1), p_var_specular], dim=1),
            }

        out = self._regress_forward(batch, vis, self.separate)
        rad_dict = {
                    'g_radiance': out['g_radiance'], 'p_radiance': out['p_radiance'],
                    'g_diffuse': out['g_diffuse'], 'p_diffuse': out['p_diffuse'],
                    'g_specular': out['g_specular'], 'p_specular': out['p_specular'],
                    }
        # kernel_dict = {
        #                 'g_kernel':out['g_kernel'],
        #                 'p_kernel': out['p_kernel'],
        #                 'new_kernel': out['new_kernel']
        #             }
        kernel_dict = None
        score_dict = {
            's_diffuse': out['s_diffuse'], 's_diffuse': out['s_diffuse'],
            's_g_diffuse': out['s_g_diffuse'], 's_p_diffuse': out['s_p_diffuse'],
            's_g_specular': out['s_g_specular'], 's_p_specular': out['s_p_specular'],
            'weight_diffuse': out['weight_diffuse'], 'weight_specular': out['weight_specular']
        }
        # kernel_dict, score_dict = None, None

        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out['radiance'],  p_buffers, rad_dict, kernel_dict, score_dict

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

class SingleStreamAdvKPCNInterface(BaseInterface):
    """
        Our adversarial method applied to a modified version of KPCN where the diffuse and specular parts are processed together in a single pass
    """
    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_adv=False, w_adv=0.0001):
        """
            Initialize SingleStreamAdvKPCNInterface
            models: (backbone, dncnn)
            optims:
            loss_funcs: (l_manif, l_recon, l_test, l_adv)
            args:
            visual=False
            use_llpm_buf=False
            manif_learn=False
            w_manif=0.1
            train_branches=True
            disentanglement_option="m11r11"
            use_skip=False
            use_adv=False
            w_adv=0.0001
        """
        # TODO remove unnecessary parameters! (disentanglement_option,)

        # check if all necessary components are given
        if manif_learn:
            assert 'backbone' in models, "argument `models` dictionary should contain `'backbone'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentanglement_option in ['m11r11', 'm10r01', 'm11r01', 'm10r11']

        # initialize
        super(SingleStreamAdvKPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.use_adv = use_adv
        self.cnt = 0
        self.epoch = 0
        self.w_adv = w_adv

    def __str__(self):
        return 'KPCNInterface'

    def to_train_mode(self):
        """
            put all models in self.models into training mode
            and check if each model has an optimizer
        """
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)

    def preprocess(self, batch=None):
        """
            check if all required components of batch are given
            and increment self.iters
            batch: contains 'target_total', 'kpcn_in', 'kpcn_buffer', and optionally 'paths'
        """
        assert 'target_total' in batch
        # TODO no target_diffuse and target_specular, right?
        assert 'kpcn_in' in batch
        assert 'kpcn_buffer' in batch
        # TODO 'kpcn_albedo' ??
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1

    def train_batch(self, batch, grad_hook_mode=False):
        """
            train SingleStreamAdvKPCNInterface with the given batch
            batch: contains 'target_total', 'kpcn_in', 'kpcn_buffer', and optionally 'paths'
            grad_hook_mode: only update model when False
        """

        out_manif = None

        if self.use_llpm_buf:
            # compute p_buffer
            self.models['backbone'].zero_grad()
            p_buffer = self._manifold_forward(batch)

            # feature disentanglement? TODO ? 
            # (just use m11r11 for now)
            out_manif = p_buffer

            # variance of p_buffer
            p_var = p_buffer.var(1).mean(1, keepdims=True).detach()
            p_var /= p_buffer.shape[1]

            # make a new batch:
            #   with extended kpcn_in (kpcn_in, p_buffer, p_var)
            #   without paths
            batch = {
                'target_total': batch['target_total'],
                'kpcn_in': torch.cat([batch['kpcn_in'], p_buffer.mean(1), p_var],1),
                'kpcn_buffer': batch['kpcn_buffer']
            }

        # KPCN
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        # compute loss
        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        # adjust model weights
        self._optimization()

    def _manifold_forward(self, batch):
        """
            compute and return p-buffer
        """
        return self.models['backbone'](batch)

    def _regress_forward(self, batch, vis=False):
        """
            process the given batch with KPCN
        """
        return self.models['dncnn'](batch,vis)

    def _backward(self, batch, out, p_buffer):
        """
            compute loss for the given input, p-buffer and output
            batch : contains 'target_total', 'kpcn_in', 'kpcn_buffer', and optionally 'paths'
            out : 'radiance', 'g_radiance', 's_f', 's_r', 'kernel', 'g_kernel'
            p_buffer
            return loss_dict : 'l_total', 'l_g_total', 'l_adv', 'l_manif', 'rmse'
        """
        assert 'radiance' in out

        # get denoised outputs
        total = out['radiance']
        g_total = out['g_radiance']

        if self.use_adv:
            # get adversarial outputs
            # TODO
            s_f_out = out['s_f']
            s_r_out = out['s_r']

        loss_dict = {}

        # get target radiance
        tgt_total = crop_like(batch['target_total'], total)

        if self.train_branches:
            # loss between total output and target
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            #loss_dict['l_total'] = L_total.detach() # TODO remove?
            L_g_total = self.loss_funcs['l_recon'](g_total, tgt_total)
            loss_dict['l_g_total'] = L_g_total.detach()

            if self.use_adv:
                # add adversarial loss
                L_adv_f = self.loss_funcs['l_adv'](s_f_out, torch.zeros_like(s_f_out))
                L_adv_r = self.loss_funcs['l_adv'](s_r_out, torch.zeros_like(s_r_out))
                L_adv = (L_adv_f + L_adv_r) / 2
                loss_dict['l_adv'] = L_adv.detach()
                L_total += L_adv * self.w_adv
            loss_dict['l_total'] = L_total.detach() 

            # joint training refinement & denoising of KPCN and manif
            if self.manif_learn:
                # compute manif loss
                p_buffer = crop_like(p_buffer, total)
                L_manif = self.loss_funcs['l_manif'](p_buffer, tgt_total)
                loss_dict['l_manif'] = L_manif.detach()

                # add to total loss
                L_total += L_manif * self.w_manif

            # backpropagate loss
            L_total.backward()

        else: # post-training the entire system
            # only compute loss between total output and target
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            loss_dict['l_total'] = L_total.detach()
            L_total.backward()

        # compute RMSE
        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](total, tgt_total).detach()

        self.cnt += 1
        return loss_dict

    def _logging(self, loss_dict):
        """
            log loss values and handle errors
        """

        # error handling
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))

        # (NOTE: modified for each model)
        for model_name in self.models:
            nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        # logging
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]

    def _optimitation(self):
        """
            optimizer step for each model in self.models
        """
        for model_name in self.models:
            self.optims['optim_' + model_name].step()

    def to_eval_mode(self):
        """
            put all models in self.models into evaluation mode
            and set m_losses zero
        """
        for model_name in self.models:
            self.modes[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)
        self.m_losses['m_gbuf_val'] = torch.tensor(0.0)

    def validate_batch(self, batch, vis=False):
        """
            validate SingleStreamAdvKPCNInterface with the given batch
        """
        p_buffer = None 

        if self.use_llpm_buf:
            # compute p_buffer
            p_buffer = self._manifold_forward(batch)

            # TODO feature disentanglement options??

            # p_buffer variance
            p_var = p_buffer.var(1).mean(1, keepdims=True).detach()
            p_var /= p_buffer.shape[1] # spp

            # make a new batch
            #   with extended kpcn_in (kpcn_in, p_buffer, p_var)
            #   without paths
            batch = {
                'target_total': batch['target_total'],
                'kpcn_in': torch.cat([batch['kpcn_in'], p_buffer.mean(1), p_var],1),
                'kpcn_buffer': batch['kpcn_buffer']
            }

        # get model output
        out = self._regress_forward(batch, vis)

        # target radiance
        tgt_total = crop_like(batch['target_total'], out['radiance'])

        # compute loss between output and target radiance
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)

        # add to m_val loss (on same device)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        # g buffer radiance loss
        L_gbuf_total = self.loss_funcs['l_test'](out['g_radiance'], tgt_total)

        # add to m_gbuf_val (on same device)
        if self.m_losses['m_gbuf_val'] == 0.0 and self.m_losses['m_gbuf_val'].device != L_gbuf_total.device:
            self.m_losses['m_gbuf_val'] = torch.tensor(0.0, device=L_gbuf_total.device)
        self.m_losses['m_gbuf_val'] += L_gbuf_total.detach()

        # prepare outputs
        rad_dict = {
            'total': out['radiance'],
            'g_total': out['g_radiance']
        }      
        kernel_dict = {
            'kernel': out['kernel'],
            'g_kernel': out['g_kernel']
        }
        score_dict = {
            'fake': out['s_f'],
            'real': out['s_r']
        }

        return out['radiance'], p_buffer, rad_dict, kernel_dict, score_dict

    def get_epoch_summary(self, mode, norm):
        """
            print epoch losses during training or return validation loss
            mode : 'train', other = evaluation
            norm : 
            return validation loss (or -1 during training)
        """
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
    
    def validate_batch(self, batch, vis=False):
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

class SingleKPCNInterface(KPCNInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11", use_skip=False, use_pretrain=False, apply_loss_twice=False):
        assert not use_llpm_buf
        assert not manif_learn
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs

        super(SingleKPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif, train_branches)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option
        self.use_skip = use_skip
        self.use_pretrain = use_pretrain
        self.apply_loss_twice = apply_loss_twice
        self.cnt = 0
        self.epoch = 0

    def __str__(self):
        return 'SingleKPCNInterface'
    
    def train_batch(self, batch):
        out_manif = None

        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        self._logging(loss_dict)

        self._optimization()
    
    def validate_batch(self, batch):
        p_buffers = None
        
        batch = {
            'target_total': batch['target_total'],
            'target_diffuse': batch['target_diffuse'],
            'target_specular': batch['target_specular'],
            'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], batch['target_diffuse']], 1),
            'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], batch['target_specular']], 1),
            'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
            'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
            'kpcn_albedo': batch['kpcn_albedo'],
        }

        out = self._regress_forward(batch)

        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out['radiance'], p_buffers


class KPCNRefInterface(KPCNInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True):
        assert not use_llpm_buf
        assert not manif_learn
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs

        super(KPCNRefInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif, train_branches)
        self.train_branches = train_branches
    
    def train_batch(self, batch):
        out_manif = None

        batch = {
            'target_total': batch['target_total'],
            'target_diffuse': batch['target_diffuse'],
            'target_specular': batch['target_specular'],
            'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], batch['target_diffuse']], 1),
            'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], batch['target_specular']], 1),
            'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
            'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
            'kpcn_albedo': batch['kpcn_albedo'],
        }

        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        self._logging(loss_dict)

        self._optimization()
    
    def validate_batch(self, batch):
        p_buffers = None
        
        batch = {
            'target_total': batch['target_total'],
            'target_diffuse': batch['target_diffuse'],
            'target_specular': batch['target_specular'],
            'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], batch['target_diffuse']], 1),
            'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], batch['target_specular']], 1),
            'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
            'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
            'kpcn_albedo': batch['kpcn_albedo'],
        }

        out = self._regress_forward(batch)

        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out['radiance'], p_buffers


class KPCNPreInterface(KPCNInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, manif_learn=False, w_manif=0.1, train_branches=True):
        # if manif_learn is True, pre-train the manifold feature extractor.
        # else, train KPCN using the freezed & pre-trained feature extractor.
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        
        use_llpm_buf = True
        super(KPCNPreInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif, train_branches)
    
    def to_train_mode(self):
        for model_name in self.models:
            if self.manif_learn:
                if 'dncnn' in model_name: # KPCN
                    self.models[model_name].eval()
                if 'backbone' in model_name: # manifold feature extractor
                    self.models[model_name].train()
            else:
                if 'dncnn' in model_name: # KPCN
                    self.models[model_name].train()
                if 'backbone' in model_name: # manifold feature extractor
                    self.models[model_name].eval()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def train_batch(self, batch):
        out_manif = None

        if self.manif_learn:
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
            
            out_manif = p_buffers

            loss_dict = self._backward(batch, None, out_manif)

            self._logging(loss_dict)

            self._optimization()
        else:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()
            self.models['dncnn'].zero_grad()

            p_buffers = self._manifold_forward(batch)
            out_manif = p_buffers

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
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

            out = self._regress_forward(batch)

            loss_dict = self._backward(batch, out, None)

            self._logging(loss_dict)

            self._optimization()
    
    def _backward(self, batch, out, p_buffers):
        assert not out or 'radiance' in out
        assert not out or 'diffuse' in out
        assert not out or 'specular' in out

        if out:
            total, diffuse, specular = out['radiance'], out['diffuse'], out['specular']
            tgt_total = crop_like(batch['target_total'], total)
        loss_dict = {}

        if self.manif_learn:
            tgt_diffuse = batch['target_diffuse'] #crop_like(batch['target_diffuse'], diffuse)
            tgt_specular = batch['target_specular'] #crop_like(batch['target_specular'], specular)

            p_buffer_diffuse = p_buffers['diffuse'] #crop_like(p_buffers['diffuse'], diffuse)
            L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse) * self.w_manif

            p_buffer_specular = p_buffers['specular'] #crop_like(p_buffers['specular'], specular)
            L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular) * self.w_manif

            loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach() / self.w_manif
            loss_dict['l_manif_specular'] = L_manif_specular.detach() / self.w_manif

            L_manif_diffuse.backward()
            L_manif_specular.backward()
        elif self.train_branches:
            tgt_diffuse = crop_like(batch['target_diffuse'], diffuse)
            L_diffuse = self.loss_funcs['l_diffuse'](diffuse, tgt_diffuse)

            tgt_specular = crop_like(batch['target_specular'], specular)
            L_specular = self.loss_funcs['l_specular'](specular, tgt_specular)

            loss_dict['l_diffuse'] = L_diffuse.detach()
            loss_dict['l_specular'] = L_specular.detach()

            L_diffuse.backward()
            L_specular.backward()

            with torch.no_grad():
                L_total = self.loss_funcs['l_recon'](total, tgt_total)
                loss_dict['l_total'] = L_total.detach()
        else:
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            loss_dict['l_total'] = L_total.detach()
            L_total.backward()

        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        for model_name in self.models:
            if self.manif_learn:
                if 'backbone' in model_name: # manifold feature extractor
                    nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)
            else:
                if 'dncnn' in model_name: # KPCN
                    nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            if self.manif_learn:
                if 'backbone' in model_name:
                    self.optims['optim_' + model_name].step()
            else:
                if 'dncnn' in model_name:
                    self.optims['optim_' + model_name].step()


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
