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


DCT_size=8
# read len RGB image and convert to grayscale
img =cv2.imread("G:/Classical Object Detection/1.jpg",0)


rows, cols = img.shape[0]//DCT_size,img.shape[1]//DCT_size
#print(rows, cols)
rows_out, cols_out = (rows*DCT_size,cols*DCT_size)
#print(rows1, cols1)
im_out=np.array([[0 for i in range(cols_out)] for j in range(rows_out)])

rows_block, cols_block = DCT_size, DCT_size

for r in range(rows):
    for c in range(cols):
        img_crop = img[r*DCT_size:(r+1)*DCT_size,c*DCT_size:(c+1)*DCT_size]
        img_DCT = dct2(img_crop)
        block=np.array([[0 for i in range(cols_block)] for j in range(rows_block)])
        for i in range(2):
            for j in range(2):
                block[i][j]=img_DCT[i][j]
        img_iDCT = idct2(block)
        im_out[r*DCT_size:(r+1)*DCT_size,c*DCT_size:(c+1)*DCT_size]=img_iDCT


print(im_out.shape)

#im1 = idct2(imF2_list)

#im1_new=imF2_list.reshape((704,1280))
#print(im1_new)

# check if the reconstructed image is nearly equal to the original image
#print(np.allclose(img, im_out))
# True

# plot original and reconstructed images with matplotlib.pylab
plt.gray()
plt.subplot(121), plt.imshow(img), plt.axis('off'), plt.title('original image', size=10)
plt.subplot(122), plt.imshow(im_out), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=10)
plt.show()