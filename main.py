import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD
import cv2

import raw_img
from LSC import LSC
from Demosaic import Demosaic
from NR import NR
from WB import WB
from CCM import CCM
from Sharpness import Sharpness
import gc

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = '4'
    path = 'RAWs/' + filename + '.dng'
    bayer_raw, bayer_type, bl, wl = raw_img.read_raw(path)  # 小米10的raw数据为10bit即最大值1023

    # 数据线性化，读出为uint16位，最小值可能会小于bl，直接减会溢出
    bayer_raw = ((np.maximum(bayer_raw, bl) - bl) / (wl - bl) * 1023).astype(np.uint16)
    tmp_im = bayer_raw[:, :, None]
    tmp_im = np.repeat(tmp_im, repeats=3, axis=2)
    matplotlib.image.imsave('./Results/' + filename + '_1_origin.jpg', tmp_im/1023)

    #  拜耳域去噪 BNR
    bnr = NR(bayer_raw)
    raw_bnr = bnr.gaussian_BNR(sigma=2, bayer_type=bayer_type)

    # Raw域白平衡
    wb = WB(bayer_raw)
    bayer_raw = wb.raw_grey_world()
    tmp_im = bayer_raw[:, :, None]
    tmp_im = np.repeat(tmp_im, repeats=3, axis=2)
    matplotlib.image.imsave('./Results/' + filename + '_2_wb.jpg', tmp_im / 1023)

    # 镜头阴影校正 LSC
    lsc = LSC(bayer_raw, bayer_type)
    bayer_raw = lsc.lsc_grid()
    tmp_im = bayer_raw[:, :, None]
    tmp_im = np.repeat(tmp_im, repeats=3, axis=2)
    matplotlib.image.imsave('./Results/' + filename + '_3_lsc.jpg', tmp_im/1023)

    #  去马赛克 Demosaic
    dm = Demosaic(bayer_raw, bayer_type)
    raw_rgb = dm.demosaic_linear()

    matplotlib.image.imsave('./Results/' + filename + '_4_demosaic.jpg', raw_rgb/1023)

    # rgb域去噪 NR
    nr = NR(raw_rgb)
    raw_rgb_nr = nr.bilateral_NR()


    # 颜色校正矩阵 CCM
    # raw_rgb_wb = np.load('./Data/raw_rgb_wb.npy')  # 加载白平衡后的图像
    ccm = CCM()
    sRGB = ccm.ccm(raw_rgb_nr)  # sRGB
    matplotlib.image.imsave('./Results/' + filename + '_5_ccm.jpg', sRGB)
    # np.save('./Data/rgb_ccm.npy', sRGB)

    # 全局gamma矫正
    nonsRGB = np.power(sRGB, 1 / 1.2)
    matplotlib.image.imsave('./Results/' + filename + '_6_gamma.jpg', nonsRGB)
    # np.save('./Data/rgb_gamma.npy', nonsRGB)

    sh = Sharpness(nonsRGB)
    sharp_RGB = sh.bilattera_sharpness()
    matplotlib.image.imsave('./Results/' + filename + '_7_sharpness.jpg', sharp_RGB)

