import cv2
from scipy import fft
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt,ceil
from pywt import dwt2

def nms(dets):
    thresh=0.5
    if len(dets) == 0:
        return []

    # Bounding boxes
    boxes = np.array(dets)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    confidence_score = boxes[:, 4]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
        
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)[::-1]

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(dets[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < thresh)
        order = order[left]
        

    return picked_boxes

def plot_portraits(image,sorted_list):
    n=ceil(sqrt(len(sorted_list)))
    n_row=n
    n_col=n
    counter=1
    plt.figure(figsize=(2.2 * n_col, 2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in sorted_list:
        plt.subplot(n_row, n_col, counter)
        plt.imshow(image[i[0]:i[2],i[1]:i[3]], cmap=plt.cm.gray)
        plt.title(i[4])
        plt.xticks(())
        plt.yticks(())
        counter+=1
    plt.show()
    

def detector(eye,normalize):
    img=cv2.imread("G:/Classical Object Detection/1.jpg",0)
    height,weight=img.shape
    dist_list=[]
    boxes=[]
    sorted_list=[]
    for k in range(-15,15,3):
        rescaled_height=int(height + (height * k/100))
        rescaled_weight=int(weight + (weight * k/100))
        resized=cv2.resize(img,(rescaled_weight,rescaled_height))
        i=1
        j=1
        while(i+64<resized.shape[0]):
            while(j+64<resized.shape[1]):
                img_croped = resized[i:i+64,j:j+64]
                img_fft=feature_extractor(img_croped,normalize)
                dist = np.linalg.norm(img_fft- eye) #euclidean distance
                '''if(i==313 and j==753):
                    sorted_list.append([i,j,i+64,j+64,dist])
                    j+=8
                    continue'''
                norm_i=int((i/resized.shape[0])*height)
                norm_j=int((j/resized.shape[1])*weight)
                dist_list.append([norm_i,norm_j,norm_i+64,norm_j+64,dist])
                j+=8           
            j=1
            i+=8
        sorted_list=sorted(dist_list, key=lambda distance: distance[4])[:10]
        #plot_portraits(img,sorted_list)

        boxes+=sorted_list
        dist_list=[]
    final_boxes=nms(boxes)
    plot_portraits(img,final_boxes)


def feature_extractor(img,normalize):

    if feature_selector==1:
        img_fft=fft.fft2(img)
        img_abs=abs(img_fft)
        img_abs[0][0]=0
        if normalize:
            img_abs=(((img_abs-img_abs.min())/(img_abs.max()-img_abs.min()))*255.0)
    
    if feature_selector==2:
        img_DCT=fft.dct(fft.dct(img.T, norm='ortho').T, norm='ortho')
        #img_abs=abs(img_DCT)
        img_abs=img_DCT
        img_abs[0][0]=0
        if normalize:
            mx=max(abs(img_abs.min()),abs(img_abs.max()))
            img_abs=img_abs/mx
            #img_abs=(((img_abs-img_abs.min())/(img_abs.max()-img_abs.min()))*255.0)

    if feature_selector==3:
        cA, (cH, cV, cD) = dwt2(img, 'haar') 
        img_abs= cA
        mx=max(abs(img_abs.min()),abs(img_abs.max()))
        img_abs=img_abs/mx
    return img_abs[:32,:32]


def rotation(degree,normalize):
    eye=cv2.imread("G:/Classical Object Detection/1.2.jpg",0)
    h,w = eye.shape
    (cX, cY) = (int(w // 2), int(h //2))
    #degreeP= npr.randint(9,15)
    MP = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated= cv2.warpAffine(eye, MP, (w, h))
    plt.figure(1)
    plt.imshow(rotated,cmap=plt.cm.gray)
    h,w = rotated.shape
    croped_eye=rotated[int(h // 2)-32:int(h // 2)+32,int(w // 2)-32:int(w // 2)+32]
    plt.figure(2)
    plt.imshow(croped_eye,cmap=plt.cm.gray)
    plt.show()
    eye_fft=feature_extractor(croped_eye,normalize)
    detector(eye_fft,normalize)

def croped_image(normalize):
    eye=cv2.imread("G:/Classical Object Detection/image.jpg",0)
    eye=cv2.resize(eye,(64,64))
    eye_fft=feature_extractor(eye,normalize)
    detector(eye_fft,normalize)



'''for i in similar:
    plt.figure()
    print(i)
    plt.imshow(img[i[0]:i[0]+64,i[1]:i[1]+64], cmap=plt.cm.gray)
    plt.show()'''

if __name__ == "__main__":
    feature_selector=3 #1=FFT , 2=DCT, 3=Wavelet
    croped_image(normalize=True)
    #rotation(degree=10,normalize=True)
