import matplotlib.pyplot as plt
import numpy as np
import os

def plot_portraits(images, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()

axa=[]
dir='G:/Classical Object Detection/ORL-Face Dataset'
celebrity_photos=os.listdir(dir)
celebrity_images=[dir+'/' + photo for photo in celebrity_photos]
for folders in celebrity_images:
    folder=os.listdir(folders)
    for images in folder:
       axa.append(folders+'/'+images)
images=np.array([plt.imread(image) for image in axa], dtype=np.float64)


n_samples, h, w = images.shape

def pca(X, n_pc):
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X-mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:,:n_pc]*S[:n_pc]
    
    return projected, components, mean, centered_data

n_components = 300
X = images.reshape(n_samples, h*w)
P, C, M, Y= pca(X, n_pc=n_components)
eigenfaces = C.reshape((n_components, h, w))
plot_portraits(eigenfaces, h, w, 4, 4) 


def reconstruction(Y, C, M, h, w, image_index):
    n_samples, n_features = Y.shape
    weights = np.dot(Y, C.T)
    centered_vector=np.dot(weights[image_index, :], C)
    recovered_image=(M+centered_vector).reshape(h, w)
    return recovered_image


recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
plot_portraits(recovered_images,h, w, n_row=4, n_col=4)