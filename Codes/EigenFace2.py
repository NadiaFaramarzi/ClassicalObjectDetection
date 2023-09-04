import os 
import numpy as np
import cv2
from matplotlib import pyplot as plt
axa=[]
dir='G:/Classical Object Detection/ORL-Face Dataset'
celebrity_photos=os.listdir(dir)
celebrity_images=[dir+'/' + photo for photo in celebrity_photos]
for folders in celebrity_images:
    folder=os.listdir(folders)
    for images in folder:
       axa.append(folders+'/'+images)
images=np.array([plt.imread(image) for image in axa], dtype=np.float64)



h=40
w=40

#print(len(images[0]))
#celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, _, _ =images.shape
images_resized=np.array([cv2.resize(image,(h,w)) for image in images], dtype=np.float64)
images_reshaped=np.array([image.reshape(h*w,1) for image in images_resized], dtype=np.float64)
#print(images_reshaped.shape)
mean=np.mean(images_reshaped,axis=0)

cov=np.array([[0 for i in range(h*w)] for j in range(h*w)])
counter=0
for i in images_reshaped:
    i=i-mean
    '''plt.imshow(i.reshape(h,w))
    plt.show()
    break'''
    vector=np.array(i)
    cov =cov+np.dot(vector,vector.T)
    counter+=1

n_pc=150
out = np.divide(cov, counter)
#eigenvalue, eigenvector = np.linalg.eig(out)
U, S, V = np.linalg.svd(out)
eigenvector = V[:n_pc]
'''plt.imshow(eigenvector[0].reshape((h, w)), cmap=plt.cm.gray)
plt.show()'''


###################################################################################

recunstruted_image=mean.T
scalar=np.dot(eigenvector,images_reshaped[10])
for i in range(len(scalar)):
    recunstruted_image=recunstruted_image+int(scalar[i])*eigenvector[i]
plt.imshow(recunstruted_image.reshape((h, w)), cmap=plt.cm.gray)
plt.show()