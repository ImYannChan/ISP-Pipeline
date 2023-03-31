# -*- encoding: utf-8 -*-
"""
@File    :   NR.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/24 14:21   CHEN      1.0         None
"""
import numpy
import cv2
import numpy as np
import cv2
import raw_img


class NR:
    def __init__(self, img):
        self.img = img.astype(np.float32)
        self.img_size = img.shape

    def gaussian_NR(self, sigma, ksize=3):
        # 手动构造高斯滤核
        # kernel_r = np.arange(0, ksize) - ksize // 2
        # kernel_l = np.arange(0, ksize) - ksize // 2
        # # 根据x，y坐标点计算高斯权重
        # kernel_list = list(map(lambda item: np.exp((-1 / (2 * alpha * alpha)) * (item[0] * item[0]) + (item[1] * item[1])),
        #                        itertools.product(kernel_r, kernel_l)))
        # kernel = (1 / 2 * np.pi * alpha) * np.array(kernel_list)
        # kernel = np.reshape(kernel / np.sum(kernel), (ksize, ksize))

        # 使用cv的高斯核函数
        kernel = cv2.getGaussianKernel(ksize=3, sigma=sigma)

        rec_im = cv2.filter2D(self.img, ddepth=-1, kernel=kernel)

        return rec_im

    def bilateral_NR(self):
        # 只支持float32和uint8
        sigmaC = 40
        sigmaS = 35
        d = 9

        dst = cv2.bilateralFilter(self.img.astype(np.float32), d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)

        return dst

    def mean_BNR(self, bayer_type='GRBG'):
        #  均值滤波
        # kernel = np.array([
        #     [1, 1, 1],
        #     [1, 1, 1],
        #     [1, 1, 1]
        # ], dtype=np.float32)/9
        R, Gr, Gb, B = raw_img.raw_channel_per(self.img, bayer_type=bayer_type)
        ksize = (5, 5)
        c11 = cv2.blur(R, ksize=ksize)
        c22 = cv2.blur(Gr, ksize=ksize)
        c33 = cv2.blur(Gb, ksize=ksize)
        c44 = cv2.blur(B, ksize=ksize)

        raw_nr = raw_img.raw_channel_combine(c11, c22, c33, c44, bayer_type=bayer_type, m_size=self.img_size)
        return raw_nr

    def gaussian_BNR(self, sigma, bayer_type):
        kernel = cv2.getGaussianKernel(ksize=7, sigma=sigma)
        R, Gr, Gb, B = raw_img.raw_channel_per(self.img, bayer_type=bayer_type)

        c11 = cv2.filter2D(R, ddepth=-1, kernel=kernel)
        c22 = cv2.filter2D(Gr, ddepth=-1, kernel=kernel)
        c33 = cv2.filter2D(Gb, ddepth=-1, kernel=kernel)
        c44 = cv2.filter2D(B, ddepth=-1, kernel=kernel)

        raw_nr = raw_img.raw_channel_combine(c11, c22, c33, c44, bayer_type=bayer_type, m_size=self.img_size)
        # minval = np.min(raw_nr, axis=None)
        # min = np.min(self.img, axis=None)
        return raw_nr

    def bilateral_BNR(self, bayer_type='GRBG'):
        # 只支持float32和uint8
        c1, c2, c3, c4 = raw_img.raw_channel_per(self.img, bayer_type=bayer_type)

        sigmaC = 15
        sigmaS = 15
        d = 5

        c11 = cv2.bilateralFilter(c1, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
        c22 = cv2.bilateralFilter(c2, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
        c33 = cv2.bilateralFilter(c3, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
        c44 = cv2.bilateralFilter(c4, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)

        raw_nr = raw_img.raw_channel_combine(c11, c22, c33, c44, bayer_type=bayer_type, m_size=self.img_size)
        return raw_nr



