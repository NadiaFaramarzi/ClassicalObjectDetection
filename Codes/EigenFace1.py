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



h=70
w=70

#print(len(images[0]))
#celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, _, _ =images.shape
images_resized=np.array([cv2.resize(image,(h,w)) for image in images], dtype=np.float64)
images_reshaped=np.array([image.reshape(h*w,1) for image in images_resized], dtype=np.float64)
#print(images_reshaped.shape)

cov=np.array([[0 for i in range(h*w)] for j in range(h*w)])
counter=0
for i in images_reshaped:
    
    average=sum(i) / len(i)
    i=i-average
    '''plt.imshow(i.reshape(h,w))
    plt.show()
    break'''
    vector=np.array(i)
    cov =cov+np.dot(vector,vector.T)
    counter+=1

n_pc=5
out = np.divide(cov, counter)
eigenvalue, eigenvector = np.linalg.eig(out)
U, S, V = np.linalg.svd(out)
eigenvector = U.T[:n_pc]
eigenv= abs(eigenvector[:][3].reshape(h,w))
eigenv= (eigenv - eigenv.min()) / (eigenv.max() - eigenv.min())
plt.imshow(np.array(abs(eigenv)), cmap=plt.cm.gray)
plt.show()