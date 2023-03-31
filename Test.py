# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import colour
import imgvision as vi

if __name__ == '__main__':
    raw_rgb_wb = np.load('Data/raw_rgb_wb.npy')

    w, h = raw_rgb_wb.shape[1], raw_rgb_wb.shape[0]
    ww, hh = w - 1160, h - 915
    raw_roi = raw_rgb_wb[950:hh, 830:ww, :]
    plt.imshow(raw_roi)
    plt.show()
    c24_rgb = []
    for i in range(4):
        for j in range(6):
            roi = raw_roi[(45 + i * 655):(660 + i * 655), 35 + j * 660: 665 + j * 660, :]
            roi = roi[100:280, 100: 280]
            mean_roi = np.reshape(roi, (-1, 3))
            c24_rgb.append(np.mean(mean_roi, axis=0))
            # plt.imshow(roi)
            # plt.show()
    c24_rgb = np.array(c24_rgb)
    np.save('./Data/c24_rgb.npy', c24_rgb )
    # xyz = np.load('./Data/c24_xyz.npy')
    # c24_srgb = colour.XYZ_to_sRGB(xyz/100.0)
    # c24_srgb[c24_srgb < 0] = 0
    # rgb = colour.sRGB_to_XYZ(c24_srgb) * 100
    # s = vi.cvtcolor()
    # rgb = s.xyz2srgb(xyz/100.0)
    # lab = s.xyz2lab(xyz)
    # np.save('./Data/c24_srgb.npy', c24_srgb)


