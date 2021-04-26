#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lossy digital image compression.

@author: khe
"""
import rawpy
import cv2
from PIL import Image
import numpy as np
from multiprocessing.pool import Pool
import utils
import os

###############################################################################
# Instantiation
###############################################################################
lum_downsample = utils.Downsampling(ratio='4:4:4')
chr_downsample = utils.Downsampling(ratio='4:2:0')
image_block = utils.ImageBlock(block_height=8, block_width=8)
dct2d = utils.DCT2D(norm='ortho')
quantization = utils.Quantization()

###############################################################################
# Preprocess
###############################################################################
# Read raw image file as array
raw = rawpy.imread(os.path.join('images', 'DSC05719.ARW'))

# Postprocess image array (Bayer filter -> RGB)
rgb_img = raw.postprocess()

# Colorspace transform (RGB -> YCrCb)
ycc_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)

# Center
ycc_img = ycc_img.astype(int)-128

# Downsampling
Y = lum_downsample(ycc_img[:,:,0])
Cr = chr_downsample(ycc_img[:,:,1])
Cb = chr_downsample(ycc_img[:,:,2])
ycc_img = np.stack((Y, Cr, Cb), axis=2)

# Create 8x8 blocks
blocks, indices = image_block.forward(ycc_img)

###############################################################################
# Compression
###############################################################################
def process_block(block, index):
    # DCT
    encoded = dct2d.forward(block)
    if index[2] == 0:
        channel_type = 'lum'
    else:
        channel_type = 'chr'
        
    # Quantization
    encoded = quantization.forward(encoded, channel_type)
    
    # Dequantization
    decoded = quantization.backward(encoded, channel_type)
    
    # Reverse DCT
    compressed = dct2d.backward(decoded)
    return compressed

compressed = np.array(Pool().starmap(process_block, zip(blocks, indices)))

###############################################################################
# Postprocess
###############################################################################
# Reconstruct image from blocks
ycc_img_compressed = image_block.backward(compressed, indices)

# Rescale
ycc_img_compressed = (ycc_img_compressed+128).astype('uint8')

# Transform back to RGB
rgb_img_compressed = cv2.cvtColor(ycc_img_compressed, cv2.COLOR_YCrCb2RGB)

# Write to file
Image.fromarray(rgb_img_compressed).save(os.path.join('images', 'result.jpeg'))