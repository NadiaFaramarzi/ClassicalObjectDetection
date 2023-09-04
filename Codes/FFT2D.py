import cv2
from scipy import fft
from matplotlib import pyplot as plt
import numpy as np


img=cv2.imread("G:/Classical Object Detection/1.jpg",0)

'''g=plt.figure(1)
plt.imshow(img,cmap='gray')
g.show()'''

fft2=fft.fft2(img)
print(fft2.min())
print(fft2.shape)
for i in range(10,700):
    for j in range(10,1270):
        fft2[i][j]=0
fft2_abs=abs(fft2)
print(fft2_abs)
fft2_abs=((fft2_abs-fft2_abs.min())/(fft2_abs.max()-fft2_abs.min()))*255.0

#print(fft2[360][600])
cv2.imshow('',fft2_abs)

ifft2 = fft.ifft2(fft2)

ifft2=((ifft2-ifft2.min())/(ifft2.max()-ifft2.min()))*255.0
ifft2=np.uint8(ifft2)
cv2.imshow('inverse',abs(ifft2))
cv2.waitKey(0)
'''h=plt.figure(3)
plt.imshow(abs(fft2),cmap='gray')
h.show()
plt.show()'''

