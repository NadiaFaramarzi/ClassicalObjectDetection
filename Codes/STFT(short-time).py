import cv2
from scipy import fft


img=cv2.imread("G:/Classical Object Detection/1.jpg",0)
print(img.shape)
fft2_list=[]
i=1
j=1
while(i+64<img.shape[0]):
    while(j+64<img.shape[1]):
        img_crop = img[i:i+64,j:j+64]
        fft2=fft.fft2(img_crop)
        fft2_abs=abs(fft2)
        fft2_abs=((fft2_abs-fft2_abs.min())/(fft2_abs.max()-fft2_abs.min()))*255.0
        fft2_list.append(fft2_abs[:4,:4])
        j+=32
    i+=32

print(fft2_list[0])
