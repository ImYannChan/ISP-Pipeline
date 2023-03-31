# -*- encoding: utf-8 -*-
"""
@File    :   Demosaic.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/23 14:36   CHEN      1.0         None
"""

import numpy as np
import cv2


class Demosaic:
    def __init__(self, bayer_raw, bayer_type):
        self.bayer_raw = bayer_raw
        self.bayer_type = bayer_type
        self.raw_size = bayer_raw.shape

    def demosaic(self):
        #  传统for循环方式实现双线性插值
        r, l = self.raw_size[0], self.raw_size[1]
        #  图像外围填充
        raw_padding = np.zeros((r + 2, l + 2))
        raw_padding[1:r + 1, 1:l + 1] = self.bayer_raw
        raw_padding[0, :] = raw_padding[2, :]
        raw_padding[r + 1, :] = raw_padding[r - 1, :]
        raw_padding[:, 0] = raw_padding[:, 2]
        raw_padding[:, l + 1] = raw_padding[:, l - 1]

        raw_rgb = np.zeros((r + 2, l + 2, 3))
        if self.bayer_type == 'GRBG':
            # R,B-> G
            for i in range(1, r + 1):
                for j in range(1, l + 1):
                    if i % 2 == 1 and j % 2 == 1:
                        raw_rgb[i, j, 1] = raw_padding[i, j]
                        #  Gr->R,
                        raw_rgb[i, j, 0] = (raw_padding[i, j - 1] + raw_padding[i, j + 1]) / 2
                        #  Gr->B
                        raw_rgb[i, j, 2] = (raw_padding[i - 1, j] + raw_padding[i + 1, j]) / 2
                    if i % 2 == 1 and j % 2 == 0:
                        raw_rgb[i, j, 0] = raw_padding[i, j]
                        #  R->Gr
                        raw_rgb[i, j, 1] = (raw_padding[i, j - 1] +
                                            raw_padding[i, j + 1] +
                                            raw_padding[i - 1, j] +
                                            raw_padding[i + 1, j]) / 4
                        # R->B
                        raw_rgb[i, j, 2] = (raw_padding[i - 1, j - 1] +
                                            raw_padding[i - 1, j + 1] +
                                            raw_padding[i + 1, j - 1] +
                                            raw_padding[i + 1, j + 1]) / 4
                    if i % 2 == 0 and j % 2 == 1:
                        raw_rgb[i, j, 2] = raw_padding[i, j]
                        # B->R
                        raw_rgb[i, j, 0] = (raw_padding[i - 1, j - 1] +
                                            raw_padding[i - 1, j + 1] +
                                            raw_padding[i + 1, j - 1] +
                                            raw_padding[i + 1, j + 1]) / 4
                        # B->G
                        raw_rgb[i, j, 1] = (raw_padding[i - 1, j] +
                                            raw_padding[i, j - 1] +
                                            raw_padding[i, j + 1] +
                                            raw_padding[i - 1, j]) / 4
                    if i % 2 == 0 and j % 2 == 0:
                        raw_rgb[i, j, 1] = raw_padding[i, j]
                        #  Gb->B
                        raw_rgb[i, j, 2] = (raw_padding[i, j - 1] + raw_padding[i, j + 1]) / 2
                        # Gb->R
                        raw_rgb[i, j, 0] = (raw_padding[i - 1, j] + raw_padding[i - 1, j]) / 2
        return raw_rgb

    def demosaic_linear(self):
        #  通过卷积方式实现双线性插值
        mask_R, mask_G, mask_B = self.__bayer_mask()

        #  卷积核
        kernel_G = np.array(
            [
                [0, 1, 0],
                [1, 4, 1],
                [0, 1, 0],
            ]) / 4

        kernel_RB = np.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]) / 4

        R = self.__conv(mask_R, kernel_RB)
        G = self.__conv(mask_G, kernel_G)
        B = self.__conv(mask_B, kernel_RB)
        raw_rgb = np.zeros((self.raw_size[0], self.raw_size[1], 3))
        raw_rgb[:, :, 0] = R
        raw_rgb[:, :, 1] = G
        raw_rgb[:, :, 2] = B
        return raw_rgb

    def __conv(self, mask, conv_kernel):
        # r, l = self.raw_size[0], self.raw_size[1]
        raw_masked = self.bayer_raw * mask
        # #  手写卷积太慢
        # raw_padding = np.zeros((r + 2, l + 2))
        # raw_padding[1:r + 1, 1:l + 1] = raw_masked
        # raw_padding[0, :] = raw_padding[2, :]
        # raw_padding[r + 1, :] = raw_padding[r - 1, :]
        # raw_padding[:, 0] = raw_padding[:, 2]
        # raw_padding[:, l + 1] = raw_padding[:, l - 1]
        # raw_channel_one = np.zeros((r + 2, l + 2))
        # for i in range(1, r + 1):
        #     for j in range(1, l + 1):
        #         data_patch = raw_padding[i - 1:i + 2, j - 1:j + 2]
        #         raw_channel_one[i, j] = np.sum(data_patch * conv_kernel, axis=None)

        #  使用opencv的卷积
        raw_conv = cv2.filter2D(raw_masked, ddepth=-1, kernel=conv_kernel)
        return raw_conv

    def __bayer_mask(self):
        R = np.zeros(self.raw_size, dtype=bool)
        G = np.zeros(self.raw_size, dtype=bool)
        B = np.zeros(self.raw_size, dtype=bool)

        if self.bayer_type == "GRBG":
            R[::2, 1::2] = True
            B[1::2, ::2] = True
            G[::2, ::2] = True
            G[1::2, 1::2] = True

        return R, G, B
