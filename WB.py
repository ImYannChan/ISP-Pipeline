# -*- encoding: utf-8 -*-
"""
@File    :   WB.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/24 15:34   CHEN      1.0         None
"""
import colour
import numpy as np
import cv2
import matplotlib.pyplot as plt

import raw_img


class WB:

    def __init__(self, img):
        self.img = img
        self.img_size = img.shape

    def raw_grey_world(self):
        R, GR, GB, B = raw_img.raw_channel_per(self.img, bayer_type='GRBG')
        G = (GR + GB) / 2
        mean_r = np.mean(R, axis=None)
        mean_g = np.mean(G, axis=None)
        mean_b = np.mean(B, axis=None)
        max_lux = np.max((mean_r, mean_g, mean_b))

        gain_r = max_lux / mean_r
        gain_g = max_lux / mean_g
        gain_b = max_lux / mean_b

        R = R * gain_r
        GR = GR * gain_g
        GB = GB * gain_g
        B = B * gain_b

        imgs = raw_img.raw_channel_combine(R, GR, GB, B, bayer_type='GRBG', m_size=self.img_size)
        imgs = np.clip(imgs, a_min=0, a_max=1023)

        return imgs.astype(np.uint16)

    def gray_world_WB(self):
        # 灰色世界算法实现白平衡
        mean_r = np.mean(self.img[:, :, 0], axis=None)
        mean_g = np.mean(self.img[:, :, 1], axis=None)
        mean_b = np.mean(self.img[:, :, 2], axis=None)
        gain_r = mean_g / mean_r
        gain_b = mean_g / mean_b
        imgs = np.zeros(self.img_size)
        imgs[:, :, 0] = self.img[:, :, 0] * gain_r
        imgs[:, :, 1] = self.img[:, :, 1]
        imgs[:, :, 2] = self.img[:, :, 2] * gain_b

        imgs = np.clip(imgs, a_min=0, a_max=1023)

        return imgs.astype(np.uint16)

    def perfect_reflect_WB(self):
        r = self.img[:, :, 0]
        g = self.img[:, :, 1]
        b = self.img[:, :, 2]

        img_add = np.sum(self.img, axis=2)
        sorted_img = np.sort(img_add.flatten(), kind='heapsort')
        ratio = 0.95  # 前5%作为白点值，阈值确定是一个重要的参数
        ind = int(len(sorted_img) * ratio)  #
        t = sorted_img[ind]
        thr_index = np.where(img_add > t)

        gr = np.max(r, axis=None) / np.mean(r[thr_index])
        gg = np.max(g, axis=None) / np.mean(g[thr_index])
        gb = np.max(b, axis=None) / np.mean(b[thr_index])

        imgs = np.zeros(self.img_size, dtype=np.uint16)
        imgs[:, :, 0] = self.img[:, :, 0] * gr
        imgs[:, :, 1] = self.img[:, :, 1] * gg
        imgs[:, :, 2] = self.img[:, :, 2] * gb
        imgs = np.clip(imgs, a_min=0, a_max=1023)

        return imgs

    def automatic_WB(self):
        # Reference：Ching-Chih Weng, H. Chen and Chiou-Shann Fuh, "A novel automatic white balance method
        # for digital still cameras," 2005 IEEE International Symposium on Circuits and Systems (ISCAS),
        # Kobe, Japan, 2005, pp. 3801-3804 Vol. 4, doi: 10.1109/ISCAS.2005.1465458.
        imgs = self.img / 1023
        yuv = self.rgb2YCrCb(imgs)
        w, h = self.img_size[1], self.img_size[0]
        y = yuv[:, :, 0]
        Cr = yuv[:, :, 1]
        Cb = yuv[:, :, 2]

        # 分块
        block_size = (3, 4)
        Mrs = []
        Mbs = []
        Drs = []
        Dbs = []
        # 计算边长
        side_x = w / block_size[1]
        side_y = h / block_size[0]
        # 对每块区域进行计算
        for i in range(block_size[0]):
            for j in range(block_size[1]):
                patch_x = np.array([j * side_x, (j + 1) * side_x], dtype=np.int)
                patch_y = np.array([i * side_y, (i + 1) * side_y], dtype=np.int)
                patch_x[patch_x > w] = w
                patch_y[patch_y > h] = h
                Cr_patch = Cr[patch_y[0]:patch_y[1], patch_x[0]:patch_x[1]]
                Cb_patch = Cb[patch_y[0]:patch_y[1], patch_x[0]:patch_x[1]]
                mr = np.mean(Cr_patch, axis=None)
                mb = np.mean(Cb_patch, axis=None)
                dr = np.mean(np.abs(Cr_patch - mr), axis=None)
                db = np.mean(np.abs(Cb_patch - mb), axis=None)
                # dr和 db太小表示该区域颜色化较小，舍去
                if dr > 0.02 and db > 0.02:
                    Mrs.append(mr)
                    Mbs.append(mb)
                    Drs.append(dr)
                    Dbs.append(db)

        Mr, Mb, Dr, Db = np.mean(Mrs), np.mean(Mbs), np.mean(Drs), np.mean(Dbs)
        # 近似白点的像素筛选
        dxs = (np.abs(Cr - (1.5 * Mr + Dr * np.sign(Mr))) < 1.5 * Dr) & \
                 (np.abs(Cb - (Mb + Db * np.sign(Mb))) < 1.5 * Db)

        y_selected = (y * dxs)
        y_s_dx = np.where(y_selected > 0)
        y_s = y_selected[y_s_dx[0], y_s_dx[1]]

        # 取白点像素集中前ratio最亮的像素
        his = np.histogram(y_s, bins=1024)
        sums = 0
        th = np.mean(y_s)
        ratio = 0.1
        for i in range(1023, 0, -1):
            sums += his[0][i]
            if sums >= len(y_s) * ratio:
                th = his[1][i]
                break
        idxs = np.where(y_selected > th)
        y_max = np.max(y_selected, axis=None)

        # RGB三通道分离
        R = imgs[:, :, 0]
        G = imgs[:, :, 1]
        B = imgs[:, :, 2]
        mean_r = np.mean(R[idxs[0], idxs[1]], axis=None)
        mean_g = np.mean(G[idxs[0], idxs[1]], axis=None)
        mean_b = np.mean(B[idxs[0], idxs[1]], axis=None)

        # 计算每个通道的增益
        gain_r = y_max / mean_r
        gain_g = y_max / mean_g
        gain_b = y_max / mean_b

        # 增益
        imgs[:, :, 0] *= gain_r
        imgs[:, :, 1] *= gain_g
        imgs[:, :, 2] *= gain_b
        np.clip(imgs * 1023, a_min=0, a_max=1023)

        return imgs.astype(np.uint16)

    # def raw_gray_world_WB(self):


    def rgb2YCrCb(self, rgb):
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        # Cr = (R - Y)  # 0.713
        # Cb = (B - Y)  # 0.564
        Cr = 0.5 * R - 0.419 * G - 0.081 * B
        Cb = -0.169 * R - 0.331 * G + 0.5 * B
        img = np.zeros(rgb.shape, dtype=np.float32)
        img[:, :, 0] = Y
        img[:, :, 1] = Cr
        img[:, :, 2] = Cb

        return img
