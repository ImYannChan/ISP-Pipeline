# -*- encoding: utf-8 -*-
"""
@File    :   raw_img.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/21 14:21   CHEN      1.0         None
"""
import numpy as np
import rawpy


def read_raw(file_path):
    raw = rawpy.imread(file_path)
    # 读取raw数据
    bayer_raw = raw.raw_image_visible.astype(np.float32)

    #获取相机的bayer阵列模式
    bayer_desc = str(raw.color_desc, encoding='utf-8')
    bayer_pattren = raw.raw_pattern.flatten()
    bayer_type = ""
    for i in bayer_pattren:
        bayer_type += bayer_desc[i]

    #  读取黑电平
    bl = raw.black_level_per_channel[0]

    #  读取饱和水平
    wl = raw.white_level

    return bayer_raw, bayer_type, bl, wl


def raw_channel_per(raw, bayer_type):
    R, Gr, Gb, B = [], [], [], []
    if bayer_type == 'RGGB':
        R = raw[::2, ::2]
        Gr = raw[::2, 1::2]
        Gb = raw[1::2, ::2]
        B = raw[1::2, 1::2]
    elif bayer_type == 'GRBG':
        Gr = raw[::2, ::2]
        R = raw[::2, 1::2]
        B = raw[1::2, ::2]
        Gb = raw[1::2, 1::2]

    return R, Gr, Gb, B


def raw_channel_combine(R, Gr, Gb, B, bayer_type, m_size):
    raw = np.zeros(m_size)
    if bayer_type == 'RGGB':
        raw[::2, ::2] = R
        raw[::2, 1::2] = Gr
        raw[1::2, ::2] = Gb
        raw[1::2, 1::2] = B
    elif bayer_type == 'GRBG':
        raw[::2, ::2] = Gr
        raw[::2, 1::2] = R
        raw[1::2, ::2] = B
        raw[1::2, 1::2] = Gb
    return raw
