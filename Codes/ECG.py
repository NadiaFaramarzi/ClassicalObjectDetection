import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

f = open("G:/ecg/ecg.txt", "r")
lines=[]
for i in f:
    lines.append(float(i.strip("\n")))
signal=np.array(lines)


vectors=[]
peaks, _ = find_peaks(signal, prominence=1)
print("peaks",len(peaks))
for i in range(len(peaks)-2):
    vectors.append(np.resize(signal[peaks[i]:peaks[i+2]],(128,)))

cov=np.array([[0 for i in range(128)] for j in range(128)])
counter=0
for i in vectors[:-5]:
    average=sum(i) / len(i)
    for j in range(len(i)):
        i[j]=i[j]-average
    vector=np.array([i])
    cov =cov+np.dot(vector.T,vector)
    counter+=1
print(counter)
 
out = np.divide(cov, counter)
eigenvalue, eigenvector = np.linalg.eig(out)
c=[]
vs=vectors[-2]-(sum(vectors[-2]) / len(vectors[-2]))

for i in range(10):
    c.append(np.dot(eigenvector.T[i],vs))

signal_new=[0 for i in range(128)]
for i in range(10):
    signal_new += c[i]*eigenvector.T[i]
 
#plt.plot(peaks, signal[peaks], "xr"); plt.plot(signal); plt.legend(['prominence'])
plt.figure(1)
plt.plot(signal_new)
plt.grid()
plt.figure(2)
plt.plot(vectors[-2])
#plt.plot(np.abs(fft),"-")
plt.grid()
plt.show()