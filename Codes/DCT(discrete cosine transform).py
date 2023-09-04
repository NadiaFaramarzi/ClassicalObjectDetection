from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

import cv2
import numpy as np
import matplotlib.pylab as plt

# read lena RGB image and convert to grayscale
im =cv2.imread("G:/Classical Object Detection/1.jpg",0)
imF = dct2(im)

rows, cols = imF.shape
imF2=np.array([[0 for i in range(cols)] for j in range(rows)])
print(imF2.shape)
for i in range(imF.shape[0]//10):
    for j in range(imF.shape[1]//10):
        imF2[i][j]=imF[i][j]

im1 = idct2(imF2)

# check if the reconstructed image is nearly equal to the original image
print(np.allclose(im, im1))
# True

# plot original and reconstructed images with matplotlib.pylab
plt.gray()
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('original image', size=10)
plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=10)
plt.show()