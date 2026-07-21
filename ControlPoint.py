import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve

DI = [0, 1, 1, 1, 0, -1, -1, -1]
DJ = [1, 1, 0, -1, -1, -1, 0, 1]
def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    height, width = src.shape[:2]

    scl_x = float(width)/maxw
    scl_y = float(height)/maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0/scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img

def unsharp(img, sigma, a):
    img = img.astype(np.float64)
    imgMF = median_filter(img, size = sigma)

    lap = cv2.Laplacian(imgMF, cv2.CV_64F, ksize = 3)

    sharp = img - a * lap
    return sharp

def blindDeconv(img, psf, iter = 50):
    deconv = np.full(img.shape, 0.1, dtype="float")
    for i in range(iter):
        psfMirror = np.flip(psf)
        conv = fftconvolve(deconv, psf, mode = "same")
        blur = img / conv

        deconv *= fftconvolve(blur, psfMirror, mode = "same")
        deconvMirror = np.flip(deconv)
        psf *= fftconvolve(blur, deconvMirror, mode = "same")
    return deconv

def rlDeblur(W, sz = 5, iter = 50):
    WNorm = W.astype(np.float64) / 255.0

    psf = cv2.getGaussianKernel(sz, sz / 3.0)
    psf = psf @ psf.T
    psf /= psf.sum()
    Wf = blindDeconv(WNorm, psf, iter)
    return np.clip(Wf * 255, 0, 255).astype(np.unit8)

def bridging(img):
    res = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype = bool)
    ans = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)

    thresh = 128 / 255.0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i+1][j+1] = img[i][j] > thresh


    for i in range(1,img.shape[0]+1):
        for j in range(1, img.shape[1]+1):
            if(res[i][j]):
                continue

            n = [res[i+DI[k]][j+DJ[k]] for k in range(8)]

            if sum(n) >= 2:
                transitions = sum(1 for k in range(8) if not n[k] and n[(k+1) % 8])
                if transitions == 2:
                    res[i][j] = True

    return res[1:-1, 1:-1]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            if(img[i][j] > thresh):
                ans[i][j] = img[i][j]
                continue

            if(not res[i+1][j+1]):
                ans[i][j] = img[i][j]
                continue
            
            cnt = 0
            sm = 0
            for k in range(8):
                ni = i + DI[k]
                nj = j + DJ[k]
                if(ni < 0 or ni >= img.shape[0] or nj < 0 or nj >= img.shape[1] or img[ni][nj] <= thresh):
                    continue


                cnt+=1
                sm += img[ni][nj]

            ans[i][j] = sm / cnt if cnt > 0 else img[i][j]

    return ans

        








img = cv2.imread("images/IMG_2987.jpg")

#img = resize_to_screen(img)

(B, G, R) = cv2.split(img)

blurred = cv2.GaussianBlur(G.astype(np.float64), (5, 5), 0)
residual = G.astype(np.float64) - blurred
n = np.std(residual)

W = G.astype(np.float64) - n
W = np.clip(W, 0, 255)

W = wiener(W, mysize=5, noise = 10)

W = np.nan_to_num(W, nan=0.0)
W = np.clip(W, 0, 255)

cv2.imshow("Before", W.astype(np.uint8))
W = unsharp(W, 5, 2)

W = np.nan_to_num(W, nan=0.0)
W = np.clip(W, 0, 255)



W = bridging(W)

W = W.astype(np.uint8) * 255



cv2.imshow("Green", W)
cv2.waitKey(0)
