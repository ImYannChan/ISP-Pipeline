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
import matplotlib.pyplot as plt

import raw_img


class LSC:
    def __init__(self, raw, bayer_type):
        self.raw = raw
        self.bayer_type = bayer_type
        self.raw_size = raw.shape
        self.path = './RAWs/white_board.dng'

    def lsc_grid(self):
        ### 使用网格法进行LSC
        R, Gr, Gb, B = raw_img.raw_channel_per(self.raw, self.bayer_type)
        try:
            #  gain_lut数据存在直接读取
            gains = np.load('Data/gain_luts.npz')
            sides_len = gains['arr_0']
            gain_luts_R = gains['arr_1']
            gain_luts_Gr = gains['arr_2']
            gain_luts_Gb = gains['arr_3']
            gain_luts_B = gains['arr_4']
            print('LSC矫正参数文件存在，直接矯正')
        except OSError:
            #  当不存在时，创建四个通道的Gain值LUT
            print('LSC矫正参数文件不存在，重新拟合')
            board_raw, bayer_type, bl, wl = raw_img.read_raw(self.path)
            board_raw = ((np.maximum(board_raw, bl) - bl) / (wl - bl) * 1023).astype(np.uint16)
            bR, bGr, bGb, bB = raw_img.raw_channel_per(board_raw, bayer_type=self.bayer_type)
            plt.figure()
            ax = plt.axes(projection="3d")
            x = np.arange(0, bR.shape[1])
            y = np.arange(0, bR.shape[0])
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, bR / 1023, cstride=40, rstride=40, cmap='jet')
            ax.plot_surface(X, Y, bGr / 1023, cstride=40, rstride=40, cmap='jet')
            ax.plot_surface(X, Y, bGb / 1023, cstride=40, rstride=40, cmap='jet')
            ax.plot_surface(X, Y, bB / 1023, cstride=40, rstride=40, cmap='jet')
            plt.show()

            block_size = (30, 26)
            sides_len = (int(bR.shape[1] / block_size[0] + 0.5),
                         int(bR.shape[0] / block_size[1] + 0.5))
            gain_luts_R = self.get_gain_lut(bR, sides=block_size)
            gain_luts_Gr = self.get_gain_lut(bGr, sides=block_size)
            gain_luts_Gb = self.get_gain_lut(bGb, sides=block_size)
            gain_luts_B = self.get_gain_lut(bB, sides=block_size)
            np.savez('./Data/gain_luts', sides_len, gain_luts_R, gain_luts_Gr, gain_luts_Gb, gain_luts_B)

        # color shading
        ratio = 0.8
        luma_gain = (gain_luts_Gr + gain_luts_Gb) / 2
        new_gain_luts_R = gain_luts_R / luma_gain
        new_gain_luts_Gr = gain_luts_Gr / luma_gain
        new_gain_luts_Gb = gain_luts_Gb / luma_gain
        new_gain_luts_B = gain_luts_B / luma_gain
        new_luma_gain = (luma_gain - 1) * ratio + 1
        new_gain_R = new_gain_luts_R * new_luma_gain
        new_gain_Gr = new_gain_luts_Gr * new_luma_gain
        new_gain_Gb = new_gain_luts_Gb * new_luma_gain
        new_gain_B = new_gain_luts_B * new_luma_gain
        #  根据gain的lut插值生成新gain

        lsc_R = self.gain_lut_interp_poly(new_gain_R, R, sides_len)
        lsc_Gr = self.gain_lut_interp_poly(new_gain_Gr, Gr, sides_len)
        lsc_Gb = self.gain_lut_interp_poly(new_gain_Gb, Gb, sides_len)
        lsc_B = self.gain_lut_interp_poly(new_gain_B, B, sides_len)
        raw_lsc = raw_img.raw_channel_combine(lsc_R, lsc_Gr, lsc_Gb, lsc_B, self.bayer_type, self.raw_size)

        raw_lsc = np.clip(raw_lsc, a_min=0, a_max=1023)
        print('LSC完成')

        return raw_lsc.astype(np.uint16)
    # 取交点的周围一块，不用外插
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

    # 取每块中心，需要外插
    def get_gain_lut(self, raw_data, sides):
        w, h = raw_data.shape[1], raw_data.shape[0]
        x_side_len, y_side_len = int(w / sides[0] + 0.5), int(h / sides[1] + 0.5)
        shade = np.zeros((sides[1], sides[0]))
        for i in range(0, sides[1]):
            for j in range(0, sides[0]):
                patch = raw_data[i * y_side_len:(i + 1) * y_side_len, j * x_side_len:(j + 1) * x_side_len]
                shade[i, j] = np.mean(patch, axis=None)
        max_shade = np.max(shade, axis=None)
        gain = max_shade / shade
        center_x, center_y = np.where(shade == max_shade)
        x, y = np.mgrid[0:sides[1], 0:sides[0]]
        Dis = np.power(x - center_x, 2) + np.power(y - center_y, 2)

        # 根据块到中心块的距离进行三次拟合
        gain_flatten = gain.flatten()
        Dis_flatten = Dis.flatten()
        A = np.polyfit(Dis_flatten, gain_flatten, 3)
        x, y = np.mgrid[0:sides[1] + 2, 0:sides[0] + 2]
        Dis_flatten = (np.power(x - center_x - 1, 2) + np.power(y - center_y - 1, 2))
        es_gain = A[0] * Dis_flatten ** 3 + A[1] * Dis_flatten ** 2 + A[2] * Dis_flatten + A[3]
        es_gain[1:sides[1]+1, 1:sides[0]+1] = gain

        return es_gain

    def gain_lut_interp_poly(self, gain_lut, raw, side_len):
        w, h = raw.shape[1], raw.shape[0]
        x_points_len = gain_lut.shape[1]
        y_points_len = gain_lut.shape[0]

        # 建立插值函数句柄
        x_line = np.linspace(0, w + side_len[0], num=x_points_len)
        y_line = np.linspace(0, h + side_len[1], num=y_points_len)
        f = sin.interp2d(x_line, y_line, gain_lut, kind='cubic')

        # 数据插值
        x_new = np.arange(side_len[0]//2, w + side_len[0]//2)
        y_new = np.arange(side_len[1]//2, h + side_len[1]//2)
        gain_interp = f(x_new, y_new)
        raw_interp = raw * gain_interp

        return raw_interp

    def gain_lut_interp_py(self, gain_lut, raw):
        w, h = raw.shape[1], raw.shape[0]
        x_points_len = gain_lut.shape[1]
        y_points_len = gain_lut.shape[0]

        x_line = np.linspace(0, w, num=x_points_len)
        y_line = np.linspace(0, h, num=y_points_len)
        f = sin.interp2d(x_line, y_line, gain_lut, kind='cubic')
        x_new = np.arange(0, w)
        y_new = np.arange(0, h)
        gain_interp = f(x_new, y_new)
        raw_interp = raw * gain_interp
        return raw_interp

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
