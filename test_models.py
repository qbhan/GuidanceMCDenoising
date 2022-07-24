import os
import re
import sys
import time
import argparse
import matplotlib.pyplot as plt 
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

import train_kpcn
import train_sbmc
import train_lbmc
import train_adv
from support.utils import crop_like
from support.img_utils import WriteImg
from support.datasets import FullImageDataset
from support.metrics import RelMSE, RelL1, SSIM, MSE, L1, _tonemap


def tonemap(c, ref=None, kInvGamma=1.0/2.2):
    # c: (W, H, C=3)
    if ref is None:
        ref = c
    luminance = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    col = np.copy(c)
    col[:,:,0] /= (1 + luminance / 1.5)
    col[:,:,1] /= (1 + luminance / 1.5)
    col[:,:,2] /= (1 + luminance / 1.5)
    col = np.clip(col, 0, None)
    return np.clip(col ** kInvGamma, 0.0, 1.0)


def load_input(filename, spp, args):    
    if 'KPCN' in args.model_name:
        dataset = FullImageDataset(filename, spp, 'kpcn',
                                   args.use_g_buf, args.use_sbmc_buf, 
                                   args.use_llpm_buf, args.pnet_out_size[0], 
                                   load_gbuf=args.load_gbuf, load_pbuf=args.load_pbuf, use_single=args.use_single)
    elif 'BMC' in args.model_name:
        dataset = FullImageDataset(filename, spp, 'sbmc',
                                   args.use_g_buf, args.use_sbmc_buf, 
                                   args.use_llpm_buf, 0)
    return dataset


def inference(interface, dataloader, spp, args):
    interface.to_eval_mode()

    H, W = dataloader.dataset.h, dataloader.dataset.w
    PATCH_SIZE = dataloader.dataset.PATCH_SIZE
    out_rad = torch.zeros((3, H, W)).cuda()
    out_rad_dict, out_kernel_dict, out_score_dict = None, None, None
    input_dict = {'diffuse_in': torch.zeros((3, H, W)).cuda(), 'specular_in': torch.zeros((3, H, W)).cuda(),}
    if args.vis_branch:
        out_rad_dict = {
            'g_radiance': torch.zeros((3, H, W)).cuda(), 'p_radiance': torch.zeros((3, H, W)).cuda(),
            'g_diffuse': torch.zeros((3, H, W)).cuda(), 'p_diffuse': torch.zeros((3, H, W)).cuda(),
            'g_specular': torch.zeros((3, H, W)).cuda(), 'p_specular': torch.zeros((3, H, W)).cuda(),
        }
    if args.kernel_visualize:
        out_kernel_dict = {
            'diffuse': torch.zeros((21*21, H, W)).cuda(),
            'specular': torch.zeros((21*21, H, W)).cuda()
        }
    if args.vis_score:
        out_score_dict = {
            's_diffuse': torch.zeros((1, H, W)).cuda(), 's_specular': torch.zeros((1, H, W)).cuda(),
            's_g_diffuse': torch.zeros((1, H, W)).cuda(), 's_p_diffuse': torch.zeros((1, H, W)).cuda(),
            's_g_specular': torch.zeros((1, H, W)).cuda(), 's_p_specular': torch.zeros((1, H, W)).cuda(),
            'weight_diffuse': torch.zeros((1, H, W)).cuda(), 'weight_specular': torch.zeros((1, H, W)).cuda(),
        }
    # if args.kernel_visualize and (args.use_skip or args.use_second_strategy or args.use_adv):
    #     out_rad_dict['g_radiance'] = torch.zeros((3, H, W)).cuda() 
    #     out_rad_dict['g_diffuse'] = torch.zeros((3, H, W)).cuda()
    #     out_rad_dict['g_specular'] = torch.zeros((3, H, W)).cuda()
    #     out_kernel_dict['g_diffuse'] = torch.zeros((21*21, H, W)).cuda()
    #     out_kernel_dict['g_specular'] = torch.zeros((21*21, H, W)).cuda()
    out_path = None
    denoise_time = 0.0
    with torch.no_grad():
        for batch, i_start, j_start, i_end, j_end, i, j in dataloader:
            for k in batch:
                if not batch[k].__class__ == torch.Tensor:
                    continue
                # batch[k] = batch[k].cuda(args.device_id)
                batch[k] = batch[k].cuda()
            start = time.time()
            out, p_buffers, rad_dict, kernel_dict, score_dict = interface.validate_batch(batch, args.kernel_visualize)
            end = time.time()
            # for k in rad_dict:
            #     print(k)
            pad_h = PATCH_SIZE - out.shape[2]
            pad_w = PATCH_SIZE - out.shape[3]
            if pad_h != 0 and pad_w != 0:
                out = nn.functional.pad(out, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters
                diffuse_in = nn.functional.pad(batch['kpcn_diffuse_in'][:,:3], (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters
                specular_in = nn.functional.pad(batch['kpcn_specular_in'][:,:3], (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters
                if args.vis_branch:
                    for k in rad_dict:
                        rad_dict[k] = nn.functional.pad(rad_dict[k], (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters
                if args.kernel_visualize:
                    for k in kernel_dict: 
                        kernel_dict[k] = nn.functional.pad(kernel_dict[k], (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters
                if args.vis_score:
                    for k in score_dict:
                        score_dict[k] = nn.functional.pad(score_dict[k], (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters
            # if args.use_llpm_buf and (out_path is None):
            #     if type(p_buffers) == dict:
            #         out_path = {}
            #         for key in p_buffers:
            #             b, s, c, h, w = p_buffers[key].shape
            #             out_path[key] = torch.zeros((s, c, H, W)).cuda()
            #     elif type(p_buffers) == torch.Tensor:
            #         b, s, c, h, w = p_buffers.shape
            #         out_path = torch.zeros((s, c, H, W)).cuda()
                # else:
                #     print('P buffer type not defined.')

            for b in range(out.shape[0]):
                out_rad[:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = out[b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                input_dict['diffuse_in'][:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = diffuse_in[b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                input_dict['specular_in'][:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = specular_in[b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                if args.vis_branch:
                    for k in rad_dict:
                        # print(k)
                        out_rad_dict[k][:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = rad_dict[k][b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                if args.kernel_visualize:                        
                    for k in kernel_dict:
                        out_kernel_dict[k][:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = kernel_dict[k][b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                if args.vis_score:
                    for k in score_dict:
                        out_score_dict[k][:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = score_dict[k][b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                # if args.use_llpm_buf:
                #     if type(p_buffers) == dict:
                #         for key in p_buffers:
                #             out_path[key][:,:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = p_buffers[key][b,:,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                    # elif type(p_buffers) == torch.Tensor:
                    #     out_path[:,:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = p_buffers[b,:,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
            denoise_time += (end - start)

    # discard log transform of specular branch
    # for k in input_dict:
    #     if 'specular' in k:
    #         input_dict[k] = torch.log(input_dict[k] - 1.0)
    # for k in out_dict:
    #     if 'specular' in k:
    #         input_dict[k] = torch.log(input_dict[k] - 1.0)

    out_rad = out_rad.detach().cpu().numpy().transpose([1, 2, 0])
    for k in input_dict:
        if 'specular' in k:
            input_dict[k] = torch.exp(input_dict[k]) - 1.0
        input_dict[k] = input_dict[k].detach().cpu().numpy().transpose([1, 2, 0])
    if args.vis_branch:
        for k in out_rad_dict:
            if 'specular' in k:
                out_rad_dict[k] = torch.exp(out_rad_dict[k]) - 1.0
            out_rad_dict[k] = out_rad_dict[k].detach().cpu().numpy().transpose([1, 2, 0])
    if args.kernel_visualize:
        for k in out_kernel_dict:
            out_kernel_dict[k] = out_kernel_dict[k].detach().cpu().numpy().transpose([1, 2, 0])
    if args.vis_score:
        for k in out_score_dict:
            out_score_dict[k] = out_score_dict[k].detach().cpu().numpy().transpose([1, 2, 0])
        # if args.use_llpm_buf:
        #     if type(out_path) == dict:
        #         for key in out_path:
        #             out_path[key] = out_path[key].detach().cpu().numpy().transpose([2, 3, 0, 1])
        #     elif type(out_path) == torch.Tensor:
        #         out_path = out_path.detach().cpu().numpy().transpose([2, 3, 0, 1])

    return out_rad, out_path, out_rad_dict, out_kernel_dict, out_score_dict, input_dict, denoise_time


def denoise(args, input_dir, output_dir="result6", scenes=None, spps=[8], save_figures=False, rhf=False, quantize=False, pixels=[(500,500)]):
    assert os.path.isdir(input_dir), input_dir
    assert 'KPCN' in args.model_name or 'BMC' in args.model_name, args.model_name

    if scenes is None:
        scenes = []
        for fn in os.listdir(input_dir.replace(os.sep + 'input', os.sep + 'gt')):
            if fn.endswith(".npy"):
                scenes.append(fn)
    # num_metrics = 5 * 4 # (RelL2, RelL1, DSSIM, L1, MSE) * (linear, tmap w/o gamma, tmap gamma=2.2, tmap gamma=adaptive)
    # results = [[0 for i in range(len(scenes))] for j in range(num_metrics * len(spps))]
    # results_input = [[0 for i in range(len(scenes))] for j in range(num_metrics * len(spps))]
    num_metrics = 1 + 6 * 4 + 1 # spp + (format, RelL2, RelL1, DSSIM, L1, MSE) * (linear, tmap w/o gamma, tmap gamma=2.2, tmap gamma=adaptive) + time
    results = [[0 for i in range(len(scenes)+1)] for j in range(num_metrics * (len(spps)))]
    # print('size1', len(results), len(results[0]))
    results_input = [[0 for i in range(len(scenes)+1)] for j in range(num_metrics * len(spps))]

    # formatting
    for i, spp in enumerate(spps):
        results[num_metrics * i][0] = str(spp)
    if args.model_name.endswith('.pth'):
        p_model = os.path.join(args.save, args.model_name)
    else:
        p_model = os.path.join(args.save, args.model_name + '.pth')
    # print('load model')
    # ck = torch.load(p_model)

    print(scenes)
    for scene in scenes:
        if not scene.endswith(".npy"):
            scene = scene + '.npy'
        filename = os.path.join(input_dir, scene).replace(os.sep + 'input', os.sep + 'gt')
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
    
    for i, scene in enumerate(scenes):
        if scene.endswith(".npy"):
            scene = scene[:-4]
        print("Scene file: ", scene)
        os.makedirs(os.path.join(output_dir, scene), exist_ok=True)

        for j, spp in enumerate(spps):
            print("Samples per pixel:", spp)
            """
            Denoising
            """
            # Dateload
            filename = os.path.join(input_dir, scene + ".npy")
            # print('load_input')
            dataset = load_input(filename, spp, args)
            # print('done')
            
            MSPP = 32# if args.pnet_out_size[0] < 12 else 8
            if spp <= MSPP:
                dataloader = DataLoader(
                    dataset,
                    batch_size=8,
                    num_workers=1
                )
            elif spp <= 64:
                dataloader = DataLoader(
                    dataset,
                    batch_size=4,
                    num_workers=1
                )
            else:
                raise RuntimeError("Try higher spp after investigating your RAM and \
                    GRAM capacity.")
            
            if i == 0 and j == 0:
                datasets = {'train': dataset} # dirty code for now
                if 'SBMC' in args.model_name:
                    interfaces, _ = train_sbmc.init_model(datasets, args)
                elif 'LBMC' in args.model_name:
                    interfaces, _ = train_lbmc.init_model(datasets, args)
                elif 'single' in args.model_name or 'adv' in args.model_name:
                    interfaces, _ = train_adv.init_model(datasets, args)
                elif 'KPCN' in args.model_name:
                    interfaces, _ = train_kpcn.init_model(datasets, args)
            '''
            if tensorrt:
                engines, contexts = export_and_load_onnx_model(interfaces[0], p_model, dataloader)
                return
            '''
            out_rad, out_path, out_rad_dict, out_kernel_dict, out_score_dict, input_dict, denoise_time = inference(interfaces[0], dataloader, spp, args)
            print('denoised time for scene {} : {:.3f} sec'.format(scene, denoise_time))
            """
            Post processing
            """
            tgt = dataset.full_tgt
            ipt = dataset.full_ipt
            # if out_path is not None:
            #     if rhf:
            #         print('Saving P-buffer as numpy file for RHF-like visualization...')
            #         if 'BMC' in args.model_name:
            #             print('Shape: ', out_path.shape)
            #             np.save(os.path.join(output_dir, 'p_buffer_%s_%s.npy'%(scene, args.model_name)), out_path)
            #         elif 'KPCN' in args.model_name:
            #             print('Shape: ', out_path['diffuse'].shape)
            #             np.save(os.path.join(output_dir, 'p_buffer_%s_%s.npy'%(scene, args.model_name)), out_path['diffuse'])
            #         print('Saved.')
            #         return
                
            #     if type(out_path) == dict:
                
            #         for key in out_path:
            #             out_path[key] = np.clip(np.mean(out_path[key], 2), 0.0, 1.0)
            #             assert len(out_path[key].shape) == 3, out_path[key].shape
            #             if out_path[key].shape[2] >= 3:
            #                 out_path[key] = out_path[key][...,:3]
            #             else:
            #                 tmp = np.mean(out_path[key], 2, keepdims=True)
            #                 out_path[key] = np.concatenate((tmp,) * 3, axis=2)
            #             assert out_path[key].shape[2] == 3, out_path[key].shape
            #     elif type(out_path) == torch.Tensor:
            #         out_path = np.clip(np.mean(out_path, 2), 0.0, 1.0)
            #         assert len(out_path.shape) == 3, out_path.shape
            #         if out_path.shape[2] >= 3:
            #             out_path = out_path[...,:3]
            #         else:
            #             tmp = np.mean(out_path, 2, keepdims=True)
            #             out_path = np.concatenate((tmp,) * 3, axis=2)
            #         assert out_path.shape[2] == 3, out_path.shape

            # Crop
            valid_size = 72
            crop = (128 - valid_size) // 2
            out_rad = out_rad[crop:-crop, crop:-crop, ...]
            for k in input_dict:
                input_dict[k] = input_dict[k][crop:-crop, crop:-crop, ...]
            if out_rad_dict:
                for k in out_rad_dict:
                    out_rad_dict[k] = out_rad_dict[k][crop:-crop, crop:-crop, ...]
            if args.kernel_visualize:
                for k in out_kernel_dict:
                    out_kernel_dict[k] = out_kernel_dict[k][crop:-crop, crop:-crop, ...]
            if out_score_dict:
                for k in out_score_dict:
                    out_score_dict[k] = out_score_dict[k][crop:-crop, crop:-crop, ...]
            if out_path is not None:
                if type(out_path) == dict:
                    for key in out_path:
                        out_path[key] = out_path[key][crop:-crop, crop:-crop, ...]
                elif type(out_path) == torch.Tensor:
                    out_path = out_path[crop:-crop, crop:-crop, ...]
            tgt = tgt[crop:-crop, crop:-crop, ...]
            ipt = ipt[crop:-crop, crop:-crop, ...]

            # Process the background and emittors which do not require to be denoised
            has_hit = dataset.has_hit[crop:-crop, crop:-crop, ...]
            out_rad = np.where(has_hit == 0, ipt, out_rad)
            if out_rad_dict:
                for k in out_rad_dict:
                    if k == 'g_radiance':
                        out_rad_dict[k] = np.where(has_hit == 0, ipt, out_rad_dict[k])
            """
            Statistics
            """
            err = RelMSE(out_rad, tgt, reduce=False)
            err = err.reshape(out_rad.shape[0], out_rad.shape[1], 3)
            
            # (RelL2, RelL1, DSSIM, L1, MSE) * (linear, tmap w/o gamma, tmap gamma=2.2, tmap gamma=adaptive)
            def linear(x):
                return x

            def tonemap28(x):
                return tonemap(x, kInvGamma = 1/2.8)

            metrics = [RelMSE, RelL1, SSIM, L1, MSE]
            tmaps = [linear, _tonemap, tonemap, tonemap28]
            metrics_str = ['RelMSE', 'RelL1', 'SSIM', 'L1', 'MSE']
            tmaps_str = ['linear', '_tonemap', 'tonemap', 'tonemap28']
            
            print(RelMSE(tonemap(out_rad), tonemap(tgt)))
            print(RelMSE(tonemap(ipt), tonemap(tgt)))

            # for t, tmap in enumerate(tmaps):
            #     for k, metric in enumerate(metrics):
            #         results[(len(metrics) * t + k) * len(spps) + j][i] = metric(tmap(out_rad), tmap(tgt))
            #         results_input[(len(metrics) * t + k) * len(spps) + j][i] = metric(tmap(ipt), tmap(tgt))
            final_index = num_metrics * (j + 1) - 1
            # for t, tmap in enumerate(tmaps):
            #     formatting = [tmaps_str[t]] + scenes
            #     results[6*t] = formatting
            #     for k, metric in enumerate(metrics):
            #         results[(6 * t + k + 1) * len(spps)][0] = metrics_str[k]
            #         results[(6 * t + k + 1) * len(spps) + j][i+1] = metric(tmap(out_rad), tmap(tgt))
            #         results_input[(6 * t + k + 1) * len(spps) + j][i+1] = metric(tmap(ipt), tmap(tgt))
            for t, tmap in enumerate(tmaps):
                formatting = [tmaps_str[t]] + scenes
                results[num_metrics * j + 6*t + 1] = formatting
                for k, metric in enumerate(metrics):
                    results[num_metrics * j + (6 * t + k + 2)][0] = metrics_str[k]
                    results[num_metrics * j + (6 * t + k + 2)][i+1] = metric(tmap(out_rad), tmap(tgt))
                    results_input[num_metrics * j + (6 * t + k + 2)][i+1] = metric(tmap(ipt), tmap(tgt))
            
            results[final_index][0] = 'time'
            results[final_index][i+1] = denoise_time

            """
            Save
            """
            if save_figures:
                if not args.load_pbuf and not 'no_pbuf' in args.model_name:
                    args.model_name += '_no_pbuf'
                if not args.load_gbuf and not 'no_gbuf' in args.model_name:
                    args.model_name += '_no_gbuf'
                t_tgt = tmaps[-1](tgt)
                t_ipt = tmaps[-1](ipt)
                t_out = tmaps[-1](out_rad)
                t_err = np.mean(np.clip(err**0.45, 0.0, 1.0), 2)

                plt.imsave(os.path.join(output_dir, scene, 'target.png'), t_tgt)
                #WriteImg(os.path.join(output_dir, scene, 'target.pfm'), tgt) # HDR image
                plt.imsave(os.path.join(output_dir, scene, 'input_{}.png'.format(spp)), t_ipt)
                #WriteImg(os.path.join(output_dir, scene, 'input_{}.pfm'.format(spp)), ipt)
                plt.imsave(os.path.join(output_dir, scene, 'output_{}_{}.png'.format(spp, args.model_name)), t_out)
                #WriteImg(os.path.join(output_dir, scene, 'output_{}_{}.pfm'.format(spp, args.model_name)), out_rad)
                plt.imsave(os.path.join(output_dir, scene, 'errmap_rmse_{}_{}.png'.format(spp, args.model_name)), t_err, cmap=plt.get_cmap('magma'))
                #WriteImg(os.path.join(output_dir, scene, 'errmap_{}_{}.pfm'.format(spp, args.model_name)), err.mean(2))
                if args.kernel_visualize:
                    print(out_score_dict['fake_diffuse'].shape)
                    plt.imsave(os.path.join(output_dir, scene, 'score_fake_diffuse_{}_{}.png'.format(spp, args.model_name)), out_score_dict['fake_diffuse'][...,0], cmap='gray')
                    plt.imsave(os.path.join(output_dir, scene, 'score_fake_specular_{}_{}.png'.format(spp, args.model_name)), out_score_dict['fake_specular'][...,0], cmap='gray')
                    plt.imsave(os.path.join(output_dir, scene, 'score_real_diffuse_{}_{}.png'.format(spp, args.model_name)), out_score_dict['real_diffuse'][...,0], cmap='gray')
                    plt.imsave(os.path.join(output_dir, scene, 'score_real_specular_{}_{}.png'.format(spp, args.model_name)), out_score_dict['real_specular'][...,0], cmap='gray')
                    # for p in pixels:
                    #     plt.imsave(os.path.join(output_dir, scene, 'crop_target_{}_{}_{}_{}.png'.format(spp, args.model_name, p[0], p[1])), t_tgt[p[0]-10:p[0]+11, p[1]-10:p[1]+11, ...])
                    #     plt.imsave(os.path.join(output_dir, scene, 'crop_input_{}_{}_{}_{}.png'.format(spp, args.model_name, p[0], p[1])), t_ipt[p[0]-10:p[0]+11, p[1]-10:p[1]+11, ...])
                    #     plt.imsave(os.path.join(output_dir, scene, 'crop_output_{}_{}_{}_{}.png'.format(spp, args.model_name, p[0], p[1])), t_out[p[0]-10:p[0]+11, p[1]-10:p[1]+11, ...])


            if args.vis_branch:
                os.makedirs(os.path.join(output_dir, scene, str(spp)), exist_ok=True)
                for k in input_dict:
                    kk = tmaps[-1](input_dict[k])
                    plt.imsave(os.path.join(output_dir, scene, str(spp), '{}_{}_{}.png'.format(spp, args.model_name,k)), kk)
                for k in out_rad_dict:
                    kk = tmaps[-1](out_rad_dict[k])
                    plt.imsave(os.path.join(output_dir, scene, str(spp), '{}_{}_{}.png'.format(spp, args.model_name,k)), kk)
            if args.kernel_visualize:
                    for p in pixels:
                        plt.imsave(os.path.join(output_dir, scene, str(spp), 'crop_{}_{}_{}_{}_{}.png'.format(spp, args.model_name,k,p[0],p[1])), kk[p[0]-10:p[0]+11, p[1]-10:p[1]+11, ...])
                    for k in out_kernel_dict:
                        for p in pixels:
                            kk = np.reshape(out_kernel_dict[k][p[0], p[1], ...], (21, 21))
                        plt.imsave(os.path.join(output_dir, scene, str(spp), 'kernel_{}_{}_{}_{}_{}.png'.format(spp, args.model_name,k,p[0],p[1])), kk, cmap='gray')
            if args.vis_score:
                for k in out_score_dict:
                    # print(out_score_dict[k].shape)
                    plt.imsave(os.path.join(output_dir, scene, str(spp), '{}_{}_{}.png'.format(spp, args.model_name,k)), out_score_dict[k][..., 0], cmap='gray')
    # x, y = len(results), len(results[0])
    # print('size', x, y)
    # for i in range(x):
    #     for j in range(y):
    #         print(results[i][j])
    np.savetxt(os.path.join(output_dir, 'results_{}_{}.csv'.format(args.model_name, spps[-1])), results, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(output_dir, 'results_input_%d.csv'%(spps[-1])), results_input, delimiter=',')


if __name__ == "__main__":
    class Args(): # just for compatibility with argparse-related functions
        output_dir = 'result_full'
        save = '/root/WCMC/weights_8/'
        model_name = 'SBMC_v2.0'
        single_gpu = True
        use_g_buf, use_sbmc_buf, use_llpm_buf = True, True, True
        
        lr_pnet = [1e-4]
        lr_ckpt = True
        pnet_out_size = [3]
        w_manif = [0.1]
        manif_learn = False
        manif_loss = 'FMSE'
        train_branches = False

        disentangle = 'm11r11'

        kpcn_ref = False

        start_epoch = 0
        single_gpu = True
        device_id = 0
        lr_dncnn = 1e-4

        visual = False
        start_epoch = 10
        best_err = 1e4

        kpcn_ref = False
        kpcn_pre = False
        not_save = False

        local = False

        # new
        use_skip = False
        use_pretrain = False
        use_second_strategy = False
        use_single = False
        use_adv = False
        separate = False
        p_depth = 2
        strided_down = False
        activation = 'relu'
        output_type = 'linear'
        disc_activation = 'relu'
        no_p_model = False
        type = None
        interpolation = 'kernel'

        # feature analysis
        load_gbuf = True
        load_pbuf = True

        # save
        kernel_visualize = False
        vis_branch = False
        vis_score = False
    
    args = Args()

    input_dir = '/mnt/ssd1/kbhan/KPCN/test/input/'
    # full scene
    scenes = ['bathroom', 'bathroom_v2', 'bathroom_v3', 'bathroom-3', 'bathroom-3_v2', 'bathroom-3_v3', 'car', 'car_v2', 'car_v3', 'car2', 'car2_v3', 'chair-room', 'chair-room_v2', 'chair', 'gharbi', 'hookah', 'hookah_v2', 'hookah_v3', 'kitchen-2', 'kitchen-2_v2', 'kitchen-2_v3', 'library-office', 'sitting-room-2', 'tableware']
    # scenes with 32, 64 spp
    # scenes = ['bathroom-3_v2', 'car', 'car_v2', 'car_v3', 'chair', 'chair-room', 'chair-room_v2', 'gharbi', 'hookah_v3', 'kitchen-2', 'kitchen-2_v2', 'library-office', 'sitting-room-2', 'tableware']
    
    
    # input_dir = '/mnt/hdd1/kbhan/KPCN/test/input/'
    # scenes = ['bathroom-2', 'bathroom', 'bedroom', 'car', 'car2', 'coffee', 'cornell-box-2', 'cornell-box', 'dining-room-2', 'gharbi', 'glass-of-water', 'kitchen', 'lamp', 'living-room-2', 'living-room-3', 'living-room', 'material-testball', 'spaceship', 'staircase-2', 'staircase']
    # scenes = ['bathroom']


    # spps = [2, 4, 8, 16]
    # spps = [32, 64]
    spps = [8]
    torch.cuda.set_device(args.device_id)
    args.save = '/home/kyubeom/WCMC/weights_8/'
    args.output_dir = 'result_8spp'

    # KPCN
    # print('KPCN')
    # args.model_name = 'KPCN_vanilla'
    # args.pnet_out_size = [0]
    # args.use_llpm_buf, args.manif_learn = False, False
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN
    # print('KPCN_manif')
    # args.model_name = 'KPCN_manifold_FMSE'
    # args.pnet_out_size = [3]
    # args.use_llpm_buf, args.manif_learn = True, True
    # args.load_gbuf, args.load_pbuf = True, True
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)
    # scenes = ['bathroom-3_v2', 'car', 'car_v2', 'car_v3', 'chair', 'chair-room', 'chair-room_v2', 'gharbi', 'hookah_v3', 'kitchen-2', 'kitchen-2_v2', 'library-office', 'sitting-room-2', 'tableware']
    # spps = [32, 64]
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN
    # print('KPCN_full')
    # args.model_name = 'KPCN_full'
    # args.pnet_out_size = [0]
    # args.use_llpm_buf, args.manif_learn = False, False
    # # spps = [2, 4, 8, 16]
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN
    # print('KPCN_manif')
    # args.model_name = 'KPCN_manif_full'
    # args.pnet_out_size = [3]
    # args.use_llpm_buf, args.manif_learn = True, True
    # args.load_gbuf, args.load_pbuf = True, True
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN_skip
    # print('KPCN_skip')
    # args.model_name = 'KPCN_manif_skip_b4_pre_finetune_new'
    # args.pnet_out_size = [3]
    # args.use_llpm_buf, args.manif_learn = True, True
    # args.use_skip, args.use_pretrain, args.use_second_strategy = True, False, False
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN_2nd
    # print('KPCN_2nd')
    # args.model_name = 'KPCN_manif_second_p_depth_4_skip'
    # args.pnet_out_size = [3]
    # args.use_llpm_buf, args.manif_learn = True, True
    # args.use_skip, args.use_pretrain, args.use_second_strategy = True, False, True
    # args.p_depth = 4
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN_adv
    # print('KPCN_adv')
    # args.model_name = 'KPCN_adv_new_w0.0001_p12'
    # args.pnet_out_size = [12]
    # args.use_llpm_buf, args.manif_learn = True, True
    # # args.use_single = True
    # args.use_adv = True
    # # args.kernel_visualize = True
    # args.kernel_visualize, args.vis_branch, args.vis_score = False, True, True
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN_new_adv_1
    print('KPCN_new_adv_1')
    args.type = 'new_adv_1'
    args.model_name = 'KPCN_new_adv_1_wadv_0.01_image_2'
    args.pnet_out_size = [12]
    args.use_llpm_buf, args.manif_learn = True, True
    args.use_adv = True
    # args.disc_activation = "leaky_relu"
    # args.kernel_visualize, args.vis_branch, args.vis_score = False, True, True
    args.interpolation = 'image'
    denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=False, output_dir=args.output_dir)

    # KPCN_adv
    # print('KPCN_adv')
    # args.model_name = 'KPCN_new_adv_2_2'
    # args.pnet_out_size = [3]
    # args.use_llpm_buf, args.manif_learn = True, False
    # args.use_single = True
    # args.use_adv = True
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # KPCN_adv
    # print('KPCN_adv')
    # args.model_name = 'KPCN_new_adv_2_disc_leaky_w0.0002'
    # args.pnet_out_size = [3]
    # args.use_llpm_buf, args.manif_learn = True, False
    # args.use_single = True
    # args.use_adv = True
    # args.disc_activation = "leaky_relu"
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # LBMC
    # print('LBMC')
    # args.model_name = 'LBMC'
    # # args.pnet_out_size = [3]
    # # args.disentangle = 'm11r11'
    # args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, False, False
    # denoise(args, input_dir, spps=[8], scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # LBMC_manif
    # print('LBMC_manif')
    # args.model_name = 'LBMC_manif_w1'
    # args.pnet_out_size = [6]
    # args.disentangle = 'm11r11'
    # args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, True, True
    # denoise(args, input_dir, spps=[8], scenes=scenes, save_figures=True, output_dir=args.output_dir)

    # SBMC
    # print('SBMC')
    # args.model_name = 'SBMC'
    # args.pnet_out_size = [0]
    # args.disentangle = 'm11r11'
    # args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = False, False, False
    # denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True, output_dir=args.output_dir)

    
    """ Test cases
    # LBMC
    print('LBMC_Path_P3')
    args.model_name = 'LBMC_Path_P3'
    args.pnet_out_size = [3]
    args.disentangle = 'm11r11'
    args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, True, False
    denoise(args, input_dir, spps=[2,4,8,16,32,64], scenes=scenes, save_figures=True)
    
    print('LBMC_Manifold_P6')
    args.model_name = 'LBMC_Manifold_P6'
    args.pnet_out_size = [6]
    args.disentangle = 'm11r11'
    args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, True, True
    denoise(args, input_dir, spps=[2,4,8,16,32,64], scenes=scenes, save_figures=True)
    
    print('LBMC_vanilla')
    args.model_name = 'LBMC_vanilla'
    args.pnet_out_size = [0]
    args.disentangle = 'm11r11'
    args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, False, False
    denoise(args, input_dir, spps=[2,4,8,16,32,64], scenes=scenes, save_figures=True)
    
    # KPCN
    print('KPCN_vanilla')
    args.model_name = 'KPCN_vanilla'
    args.pnet_out_size = [0]
    args.use_llpm_buf, args.manif_learn = False, False
    denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True)
    
    print('KPCN_path')
    args.model_name = 'KPCN_path'
    args.pnet_out_size = [3]
    args.disentangle = 'm11r11'
    args.use_llpm_buf, args.manif_learn = True, False
    denoise(args, input_dir, spps=spps, scenes=scenes, rhf=True)
     
    # SBMC
    print('SBMC_vanilla')
    args.model_name = 'SBMC_vanilla'
    args.pnet_out_size = [0]
    args.disentangle = 'm11r11'
    args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, False
    denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True)

    print('SBMC_path')
    args.model_name = 'SBMC_path'
    args.pnet_out_size = [3]
    args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = False, True, False
    denoise(args, input_dir, spps=spps, scenes=scenes, rhf=True)
    
    print('SBMC_Manifold_Naive')
    args.model_name = 'SBMC_Manifold_Naive'
    args.pnet_out_size = [3]
    args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = False, True, False
    denoise(args, input_dir, spps=spps, scenes=scenes)
    """
