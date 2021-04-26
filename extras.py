#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The slow implementation of some utility functions using loops. The code here is
purely written for explict demonstration and NOT for performance.  

@author: khe
"""
import numpy as np

def dct2d(x):
    # Normalizing scalar
    def alpha(u):
        return 1/(2**0.5) if u == 0 else 1
    
    out = np.zeros((x.shape))
    for u in range(out.shape[0]):
        for v in range(out.shape[1]):
            scalar = 0.25*alpha(u)*alpha(v)
            val = 0.0
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    val += x[i,j]*np.cos((2*i+1)*u*np.pi/16)*np.cos((2*j+1)*v*np.pi/16)
            out[u,v] = scalar*val
    return out

def idct2d(x):
    # Normalizing scalar
    def alpha(u):
        return 1/(2**0.5) if u == 0 else 1
    
    out = np.zeros((x.shape))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            scalar = 0.25
            val = 0.0
            for u in range(x.shape[0]):
                for v in range(x.shape[1]):
                    val += alpha(u)*alpha(v)*x[u,v]*np.cos((2*i+1)*u*np.pi/16)*np.cos((2*j+1)*v*np.pi/16)
            out[i,j] = scalar*val
    return out

def downsampling(x, ratio='4:2:0'):
    assert ratio in ('4:4:4', '4:2:2', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:2', '4:2:0'}"
    # No subsampling
    if ratio == '4:4:4':
        return x
    else:
        out = np.zeros((x.shape))
        # Downsample with a window of 2 in the horizontal direction
        if ratio == '4:2:2':
            for i in range(0, x.shape[0], 2):
                out[i:i+2] = np.mean(x[i:i+2], axis=0)
        # Downsample with a window of 2 in both directions
        else:
            for i in range(0, x.shape[0], 2):
                for j in range(0, x.shape[1], 2):
                    out[i:i+2, j:j+2] = np.mean(x[i:i+2, j:j+2])
        return np.round(out).astype('uint8')