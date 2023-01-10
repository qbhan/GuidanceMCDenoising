import numpy as np
import torch
import torch.nn as nn


def crop_like(src, tgt):
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    # delta = (src_sz[2:4]-tgt_sz[2:4])
    delta = (src_sz[-2:]-tgt_sz[-2:])
    crop = np.maximum(delta // 2, 0)  # no negative crop
    crop2 = delta - crop

    if (crop > 0).any() or (crop2 > 0).any():
        # NOTE: convert to ints to enable static slicing in ONNX conversion
        src_sz = [int(x) for x in src_sz]
        crop = [int(x) for x in crop]
        crop2 = [int(x) for x in crop2]
        return src[..., crop[0]:src_sz[-2]-crop2[0],
                   crop[1]:src_sz[-1]-crop2[1]]
    else:
        return src

def ToneMap(c, limit=1.5):
    # c: (W, H, C=3)
    luminance = 0.2126 * c[:,:,0] + 0.7152 * c[:,:,1] + 0.0722 * c[:,:,2]
    col = c.copy()
    col[:,:,0] /=  (1.0 + luminance / limit)
    col[:,:,1] /=  (1.0 + luminance / limit)
    col[:,:,2] /=  (1.0 + luminance / limit)
    return col

def LinearToSrgb(c):
    # c: (W, H, C=3)
    kInvGamma = 1.0 / 2.2
    return np.clip(c ** kInvGamma, 0.0, 1.0)

def ToneMapBatch(c):
    # originally for numpy
    # c: (B, C=3, W, H)
    # luminance = 0.2126 * c[:,0,:,:] + 0.7152 * c[:,1,:,:] + 0.0722 * c[:,2,:,:]
    # # col = c.copy()
    # col = c.clone()
    # col[:,0,:,:] /= (1 + luminance / 1.5)
    # col[:,1,:,:] /= (1 + luminance / 1.5)
    # col[:,2,:,:] /= (1 + luminance / 1.5)
    # col = torch.clip(col, 0, None)
    kInvGamma = 1.0 / 2.2
    col = torch.clamp(c, min=0)
    col = col / (1 + col)
    # return torch.clip(col ** kInvGamma, 0.0, 1.0)
    return col ** kInvGamma