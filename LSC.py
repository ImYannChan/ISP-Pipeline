# -*- encoding: utf-8 -*-
"""
@File    :   LSC.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/21 21:19   CHEN      1.0         None
"""
import numpy as np
import scipy.interpolate as sin

import raw_img


class LSC:
    def __init__(self, raw, bayer_type):
        self.raw = raw
        self.bayer_type = bayer_type
        self.raw_size = raw.shape
        self.path = './RAWs/white_board.dng'

    def lsc_grid(self):
        ### 使用网格法进行LSC
        # R, Gr, Gb, B = raw_img.raw_channel_per(self.raw, self.bayer_type)
        try:
            #  gain_lut数据存在直接读取
            gain_luts = np.load('Data/gain_luts.npy').astype(np.float32)
            print('LSC矫正参数文件存在，直接矯正')
        except OSError:
            #  当不存在时，创建四个通道的Gain值LUT
            print('LSC矫正参数文件不存在，重新拟合')
            board_raw, bayer_type, bl, wl = raw_img.read_raw(self.path)
            board_raw = np.maximum(board_raw - bl, 0) / (wl - bl)
            gain_luts = self.create_gain_lut(board_raw, sides=(24, 18))

            np.save('./Data/gain_luts', gain_luts)

        #  根据gain的lut插值生成新gain
        print('开始校正')
        raw_lsc = self.gain_lut_interp_py(gain_luts, self.raw)
        print('校正完成')
        return raw_lsc

    def create_gain_lut(self, raw_data, sides):
        x_sides, y_sides = sides[0], sides[1]
        w, h = raw_data.shape[1], raw_data.shape[0]
        # LUT点
        x_line = np.linspace(0, w, num=x_sides + 1)
        y_line = np.linspace(0, h, num=y_sides + 1)
        # 计算边长
        dx = np.diff(x_line)
        dy = np.diff(y_line)
        gain_lut = np.zeros((y_sides + 1, x_sides + 1))
        #  计算每个点周围小块内的平均值
        for i in range(0, len(y_line)):
            for j in range(0, len(x_line)):
                x = j % len(dx)
                y = i % len(dy)
                x_patch = np.array([x_line[j] - dx[x] / 2, x_line[j] + dx[x] / 2],
                                   dtype=np.int)
                y_patch = np.array([y_line[i] - dy[y] / 2, y_line[i] + dy[y] / 2],
                                   dtype=np.int)
                x_patch[x_patch < 0] = 0
                x_patch[x_patch > w] = w
                y_patch[y_patch < 0] = 0
                y_patch[y_patch > h] = h

                data_patch = raw_data[y_patch[0]:y_patch[1], x_patch[0]:x_patch[1]]
                gain_lut[i, j] = np.mean(data_patch)
        #  计算gain值
        max_mean = np.max(gain_lut)
        for i in range(0, len(y_line)):
            for j in range(0, len(x_line)):
                gain_lut[i, j] = max_mean / gain_lut[i, j]
        
        return gain_lut

    def gain_lut_interp_py(self, gain_lut, raw):
        w, h = raw.shape[1], raw.shape[0]
        x_points_len = gain_lut.shape[1]
        y_points_len = gain_lut.shape[0]

        x_line = np.linspace(0, w, num=x_points_len)
        y_line = np.linspace(0, h, num=y_points_len)
        f = sin.interp2d(x_line, y_line, gain_lut, kind='linear')
        x_new = np.arange(0, w)
        y_new = np.arange(0, h)
        gain_interp = f(x_new, y_new)
        raw_interp = raw * gain_interp
        raw_interp[raw_interp > 1] = 1.0
        return raw_interp.astype(np.float32)

    def gain_lut_interp(self, gain_lut, raw):
        #  根据创建的GAIN_LUT对原raw图像进行插值校正
        w, h = raw.shape[1], raw.shape[0]
        x_points_len = gain_lut.shape[1]
        y_points_len = gain_lut.shape[0]
        side_x = w / (x_points_len - 1)
        side_y = h / (y_points_len - 1)

        for i in range(0, h):
            for j in range(0, w):
                x = int(j // side_x)
                y = int(i // side_y)
                u = j / side_x - x
                v = i / side_y - y
                # 根据最近四点进行双线性插值
                # f[a, b] = (1 - u) * (1 - v) * f[x, y] + u * (1 - v) * f[x + 1, y] + \
                #          (1 - u) * v * f[x, y + 1] + u * v * f[x + 1, y + 1]
                gain = (1 - u) * (1 - v) * gain_lut[x, y] + \
                       u * (1 - v) * gain_lut[x + 1, y] + \
                       (1 - u) * v * gain_lut[x, y + 1] + \
                       u * v * gain_lut[x + 1, y + 1]
                raw[i, j] *= gain

        return raw
