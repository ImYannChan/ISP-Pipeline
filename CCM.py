# -*- encoding: utf-8 -*-
"""
@File    :   CCM.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/25 15:40   CHEN      1.0         None
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as sopt
import colour
import gc


class CCM:
    ### 此处的CCM使用的是D65光源下的颜色进行矫正，采用多项式扩展的方式
    def __init__(self):

        self.fit_data = np.load('./Data/c24_rgb.npy')
        self.target_data = np.load('./Data/c24_xyz.npy') / 100.0  # 归一化到0-1
        self.degree = 2
        try:
            self.CMatrix = np.load('Data/CCM.npy')
            print("颜色矫正矩阵文件存在，直接矫正")

        except OSError:
            print("颜色矫正矩阵文件不存在，重新拟合颜色矫正矩阵...")
            self.CMatrix = []  # 储存最终的CCM的矩阵
            self.fit()
            print("拟合完成...")

    def ccm(self, img):
        img_shape = img.shape
        img_f = np.reshape(img, (-1, 3))
        predict_sRGB = self.predict(img_f)

        predict_sRGB = np.reshape(predict_sRGB, img_shape)
        return predict_sRGB

    def __read_data(self, path):
        #  考虑读取其他文件时进行扩展
        data = np.load(path)
        return data

    def fit(self):

        #
        fit_data_poly = self.fit_data
        # 多项式扩展后的项数
        # fit_data_poly = PolynomialFeatures(degree=self.degree).fit_transform(self.fit_data)
        term_num = fit_data_poly.shape[1]
        # 初始化点
        matrix0 = np.linalg.pinv(fit_data_poly).dot(self.target_data)
        #  计算target_xyz和lab
        target_lab = colour.XYZ_to_Lab(self.target_data)  # 默认D65光源
        #  从sRGB转lab
        # target_XYZ = colour.sRGB_to_XYZ(self.target_data)
        # target_lab = colour.XYZ_to_Lab(target_XYZ)

        # 构造损失函数
        def loss_fun(x):
            matrix = np.reshape(x, (term_num, 3))

            ### 此处考虑直接求raw_rgb -> sRGB的转换矩阵，暂时没调试
            # fit_srgb = np.dot(fit_data_poly, matrix)
            # fit_xyz = colour.sRGB_to_XYZ(fit_srgb)
            # fit_lab = colour.XYZ_to_Lab(fit_xyz)
            ### ...................................#####

            # 乘以矫正矩阵
            fit_xyz = np.dot(fit_data_poly, matrix)
            # 计算fit_lab
            fit_lab = colour.XYZ_to_Lab(fit_xyz)  # 默认D65光源

            # 计算平均色差，作为损失函数,可以考虑使用CIEDE2000色差公式
            mean_de = np.mean(np.sum((fit_lab - target_lab) ** 2, axis=1))
            return mean_de

        #  最优化
        print('开始拟合CCM矩阵')
        x0 = matrix0.flatten()
        # 约束条件
        cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[3] + x[6] - x[1] - x[4] - x[7]},
                {'type': 'eq', 'fun': lambda x: x[1] + x[4] + x[7] - x[2] - x[5] - x[8]})
        # fit_x, _ = sopt.leastsq(loss_fun, x0)
        fit_x = sopt.minimize(loss_fun, x0, method='BFGS').x  # 非线性规划

        self.CMatrix = np.reshape(fit_x, (term_num, 3)).astype(np.float32)
        np.save('Data/CCM.npy', self.CMatrix)
        print('拟合完成')

    def predict(self, test_data):
        # test_data_poly = PolynomialFeatures(degree=self.degree).fit_transform(test_data)

        test_data_poly = test_data
        predict_xyz = np.dot(test_data_poly, self.CMatrix)
        # predict_sRGB = np.dot(test_data_poly, self.CMatrix)

        #  CIE XYZ转线性sRGB,此处因为内存问题分为了两部分进行计算
        end = predict_xyz.shape[0] // 2
        predict_sRGB_1 = colour.XYZ_to_sRGB(predict_xyz[:end, :])
        predict_sRGB_2 = colour.XYZ_to_sRGB(predict_xyz[end:, :])
        predict_sRGB = np.vstack((predict_sRGB_1, predict_sRGB_2))

        # 会有超色域的点，规范到0-1
        predict_sRGB[predict_sRGB > 1] = 1.0
        predict_sRGB[predict_sRGB < 0] = 0
        del predict_xyz
        gc.collect()
        return predict_sRGB


if __name__ == '__main__':

    sd = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]
    illuminant = colour.SDS_ILLUMINANTS["D65"]
    xyz = []
    for key in sd.keys():
        spec = sd[key]
        a = colour.sd_to_XYZ(spec, illuminant=illuminant)
        xyz.append(a)
    c24_xyz = np.array(xyz)
    np.save('./Data/c24_xyz.npy', c24_xyz)
    # plot_single_colour_checker("ColorChecker 2005", text_kwargs={"visible": False})
