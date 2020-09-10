import numpy as np
import cv2
from colour_demosaicing import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_bilinear
import math
import random
import scipy.io as scio


class In_Camera_Process():

    def __init__(self, sigma_s=None, sigma_c=None, crf_index=None, pattern=None, mode='Train', corr=False):
        self.data_I_B = scio.loadmat('./simulation/201_CRF_data.mat')
        self.data_invI_invB = scio.loadmat('./simulation/dorfCurvesInv.mat')
        self.sigma_s = sigma_s
        self.sigma_c = sigma_c
        self.crf_index = crf_index
        self.pattern = pattern
        self.mode = mode
        self.corr = corr

    def ICRF_Map(self, img, invI, invB):
        w, h, c = img.shape
        bin = len(invI)
        tiny_bin = 9.7656e-04
        min_tiny_bin = 0.0039
        out = []
        img = img.tolist()
        for i in range(0, w):
            for j in range(0, h):
                for k in range(0, c):
                    temp = img[i][j][k]
                    pixel = temp
                    start_bin = 0
                    if temp > min_tiny_bin:
                        start_bin = math.floor(temp / tiny_bin - 1) - 1
                    for b in range(start_bin, bin):
                        tempB = invB[b]
                        if tempB >= temp:
                            index = b
                            if index > 0:
                                comp1 = tempB - temp
                                comp2 = temp - invB[index - 1]
                                if comp2 < comp1:
                                    index = index - 1
                            pixel = invI[index]
                            break
                    out.append(pixel)
        return np.array(out, dtype=np.float32).reshape((w, h, c))

    def CRF_Map(self, img, I, B):
        w, h, c = img.shape
        bin = len(I)
        tiny_bin = 9.7656e-04
        min_tiny_bin = 0.0039
        out = []
        img = img.tolist()
        for i in range(0, w):
            for j in range(0, h):
                for k in range(0, c):
                    temp = img[i][j][k]
                    if temp < 0:
                        temp = 0
                    if temp > 1:
                        temp = 1
                    pixel = temp
                    start_bin = 0
                    if temp > min_tiny_bin:
                        start_bin = math.floor(temp / tiny_bin - 2)
                    for b in range(start_bin, bin):
                        tempB = I[b]
                        if tempB >= temp:
                            index = b
                            if index > 0:
                                comp1 = tempB - temp
                                comp2 = temp - B[index - 1]
                                if comp2 < comp1:
                                    index = index - 1
                            pixel = B[index]
                            break
                    out.append(pixel)
        return np.array(out, dtype=np.float32).reshape((w, h, c))

    def synthesize_noise(self, img, max_s=25/255, max_c=25/255, min_s=0, min_c=0):
        channel = img.shape[2]  # W H C: rgb
        if self.sigma_s is None:
            np.random.seed(seed=None)
            sigma_s = np.random.uniform(min_s, max_s, (1, 1, channel))
        else:
            sigma_s = self.sigma_s
        if self.sigma_c is None:
            np.random.seed(seed=None)
            sigma_c = np.random.uniform(min_c, max_c, (1, 1, channel))
        else:
            sigma_c = self.sigma_c
        if self.crf_index is None:
            np.random.seed(seed=None)
            crf_index = random.randint(0, 200)
        else:
            crf_index = self.crf_index
        if self.pattern is None:
            np.random.seed(seed=None)
            pattern = random.randint(0, 5)
        else:
            pattern = self.pattern

        I = self.data_I_B['I'][crf_index, :].tolist()
        B = self.data_I_B['B'][crf_index, :].tolist()
        invI = self.data_invI_invB['invI'][crf_index, :].tolist()
        invB = self.data_invI_invB['invB'][crf_index, :].tolist()

        # x-->L
        temp_x = self.ICRF_Map(img, invI, invB)

        # adding noise
        noise_s_map = np.tile(sigma_s, (temp_x.shape[0], temp_x.shape[1], 1)) * temp_x
        if self.mode == 'Test':
            np.random.seed(seed=0)  # for reproducibility
            noise_s = np.random.normal(0, 1, temp_x.shape) * noise_s_map
        else:
            np.random.seed(seed=None)
            noise_s = np.random.normal(0, 1, temp_x.shape) * noise_s_map

        noise_c_map = np.tile(sigma_c, (temp_x.shape[0], temp_x.shape[1], 1))
        if self.mode == 'Test':
            np.random.seed(seed=0)  # for reproducibility
            noise_c = np.random.normal(0, 1, temp_x.shape) * noise_c_map
        else:
            np.random.seed(seed=None)
            noise_c = np.random.normal(0, 1, temp_x.shape) * noise_c_map

        temp_n = temp_x + noise_s + noise_c
        noise_map = np.sqrt(noise_s_map + noise_c_map)

        if self.corr:
            # L-->x
            temp_x = self.CRF_Map(temp_x, I, B)

            # add Mosai
            if pattern == 0:
                B_b_x = mosaicing_CFA_Bayer(temp_x, 'GBRG')
            elif pattern == 1:
                B_b_x = mosaicing_CFA_Bayer(temp_x, 'GRBG')
            elif pattern == 2:
                B_b_x = mosaicing_CFA_Bayer(temp_x, 'BGGR')
            elif pattern == 3:
                B_b_x = mosaicing_CFA_Bayer(temp_x, 'RGGB')
            else:
                B_b_x = temp_x
            temp_x = B_b_x

            # DeMosai
            if pattern == 0:
                lin_rgb_x = demosaicing_CFA_Bayer_bilinear(temp_x, 'GBRG')
            elif pattern == 1:
                lin_rgb_x = demosaicing_CFA_Bayer_bilinear(temp_x, 'GRBG')
            elif pattern == 2:
                lin_rgb_x = demosaicing_CFA_Bayer_bilinear(temp_x, 'BGGR')
            elif pattern == 3:
                lin_rgb_x = demosaicing_CFA_Bayer_bilinear(temp_x, 'RGGB')
            else:
                lin_rgb_x = temp_x
            temp_x = lin_rgb_x


        # L-->x
        temp_n = self.CRF_Map(temp_n, I, B)

        # add Mosai
        if pattern == 0:
            B_b_n = mosaicing_CFA_Bayer(temp_n, 'GBRG')
        elif pattern == 1:
            B_b_n = mosaicing_CFA_Bayer(temp_n, 'GRBG')
        elif pattern == 2:
            B_b_n = mosaicing_CFA_Bayer(temp_n, 'BGGR')
        elif pattern == 3:
            B_b_n = mosaicing_CFA_Bayer(temp_n, 'RGGB')
        else:
            B_b_n = temp_n
        temp_n = B_b_n

        # DeMosai
        if pattern == 0:
            lin_rgb_n = demosaicing_CFA_Bayer_bilinear(temp_n, 'GBRG')
        elif pattern == 1:
            lin_rgb_n = demosaicing_CFA_Bayer_bilinear(temp_n, 'GRBG')
        elif pattern == 2:
            lin_rgb_n = demosaicing_CFA_Bayer_bilinear(temp_n, 'BGGR')
        elif pattern == 3:
            lin_rgb_n = demosaicing_CFA_Bayer_bilinear(temp_n, 'RGGB')
        else:
            lin_rgb_n = temp_n
        temp_n = lin_rgb_n

        if self.corr:
            y = temp_n - temp_x + img
        else:
            y = temp_n
        return y, noise_map


if __name__ == "__main__":

    def main():
        import matplotlib.pyplot as plt
        from skimage.measure import compare_psnr, compare_ssim
        img = cv2.imread('simulation/kodim23.png', cv2.IMREAD_COLOR).astype(np.float32)  # W H C
        img = img[:, :, ::-1] / 255  # from BGR to RGB
        icp = In_Camera_Process(mode='Train',
                                sigma_c=np.array([[[0.06, 0.06, 0.06]]]),
                                sigma_s=np.array([[[0.03, 0.03, 0.03]]]),
                                corr=True
                                )
        img_n, nlm = icp.synthesize_noise(img, 0.16, 0.06)

        psnr = round(compare_psnr(img_n, img, data_range=1), 3)
        print(psnr)

        nlm = np.sqrt(np.mean(nlm**2, 2))
        nlm = np.uint8(nlm * 255)

        plt.subplot(1, 3, 1)
        plt.title('origin image')
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('synthetic noisy image')
        plt.imshow(img_n)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('nlm')
        plt.imshow(nlm)
        plt.axis('off')
        plt.show()
    main()
