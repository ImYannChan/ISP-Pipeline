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
        dst[dst > 1] = 1.0
        dst[dst < 0] = 0
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


if __name__ == '__main__':
    img = np.load('Data/rgb_gamma.npy')
    sh = Sharpness(img)
    rgb = sh.sharpness()
    plt.imshow(rgb)
    plt.show()
    matplotlib.image.imsave('./Results/3_Sharpness.jpg', rgb)
