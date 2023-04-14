# -*- encoding: utf-8 -*-
"""
@File    :   Sharpness.py    
@Contact :   chen_yang0921@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/5/26 19:36   CHEN      1.0         None
"""
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Sharpness:
    def __init__(self, img):
        self.img = img.astype(np.float32)

    def sharpness(self):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        dst = cv2.filter2D(self.img, ddepth=-1, kernel=kernel)
        dst = np.clip(dst, a_min=0, a_max=1.0)

        return dst

    def heq(self):
        # 彩色图像直方图均衡化。转化到YCrCb颜色空间后进行处理
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0]
        histy, binsy = np.histogram(y.flatten(), bins=256, range=(0, 1))

        cdfy = np.cumsum(histy)
        cdfy = cdfy / cdfy[-1]

        y_heq = np.interp(y.flatten(), xp=binsy[:-1], fp=cdfy)
        y = np.reshape(y_heq, y.shape)
        ycrcb[:, :, 0] = y
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        rgb[rgb > 1] = 1.0
        rgb[rgb < 0] = 0

        return rgb

    def bilattera_sharpness(self):
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0]

        # 减去模糊图，得到边缘图
        sigmaC = 95
        sigmaS = 95
        d = 7
        blur_Y = cv2.bilateralFilter(y, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
        diff_y = y - blur_Y

        # 加权融合原图和边缘图
        w = 1.2
        new_y = y + w * diff_y
        ycrcb[:, :, 0] = new_y
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        rgb = np.clip(rgb, a_min=0, a_max=1.0)

        return rgb

    def gaussian_sharpness(self):
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0]
        plt.imshow(y, cmap='gray')
        plt.show()

        # 减去模糊图，得到边缘图
        sigma = 2.0
        ksize = (5, 5)
        edge_Y = cv2.GaussianBlur(y, ksize=ksize, sigmaX=sigma)
        diff_y = y - edge_Y
        plt.imshow(diff_y, cmap='gray')
        plt.show()

        # 加权融合原图和边缘图
        w = 1.5
        new_y = y + w * diff_y
        ycrcb[:, :, 0] = new_y
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        plt.imshow(rgb)
        plt.show()

        return rgb


if __name__ == '__main__':
    img = np.load('Data/rgb_gamma.npy')
    sh = Sharpness(img)
    rgb = sh.sharpness()
    plt.imshow(rgb)
    plt.show()
    matplotlib.image.imsave('./Results/3_Sharpness.jpg', rgb)
