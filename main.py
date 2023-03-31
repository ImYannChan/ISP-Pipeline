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
    filename = '3'
    path = 'RAWs/' + filename + '.dng'
    bayer_raw, bayer_type, bl, wl = raw_img.read_raw(path)  # 小米10的raw数据为10bit即最大值1023
    # 数据线性化
    bayer_raw = (np.maximum(bayer_raw - bl, 0) / (wl - bl)).astype(np.float32)

    #  拜耳域去噪 BNR
    bnr = NR(bayer_raw)
    raw_bnr = bnr.gaussian_BNR(sigma=0.8, bayer_type=bayer_type)

    # 镜头阴影校正 LSC
    lsc = LSC(raw_bnr, bayer_type)
    bayer_raw = lsc.lsc_grid()

    #  去马赛克 Demosaic
    dm = Demosaic(bayer_raw, bayer_type)
    raw_rgb = dm.demosaic_linear()

    # 去噪 NR
    nr = NR(raw_rgb)
    raw_rgb_nr = nr.bilateral_NR()

    #  白平衡 WB
    wb = WB(raw_rgb)
    raw_rgb_wb = wb.gray_world_WB()
    # plt.imshow(raw_rgb_wb)
    # plt.show()
    # np.save('./Data/raw_rgb_wb', raw_rgb_wb)
    matplotlib.image.imsave('./Results/' + filename + '_wb.jpg', raw_rgb_wb)
    del bnr, lsc, dm, nr, wb
    gc.collect()

    # 颜色校正矩阵 CCM
    raw_rgb_wb = np.load('./Data/raw_rgb_wb.npy')  # 加载白平衡后的图像
    ccm = CCM()
    sRGB = ccm.ccm(raw_rgb_wb)  # 线性sRGB

    matplotlib.image.imsave('./Results/' + filename + '_ccm.jpg', sRGB)
    # np.save('./Data/rgb_ccm.npy', sRGB)

    # 全局gamma矫正
    nonsRGB = np.power(sRGB, 1 / 1.3)
    matplotlib.image.imsave('./Results/' + filename + '.jpg', nonsRGB)
    # np.save('./Data/rgb_gamma.npy', nonsRGB)


