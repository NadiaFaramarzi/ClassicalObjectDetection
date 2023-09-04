import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.fft import fft,ifft


signal=np.array([i for i in range(1,51)])
print(signal)
ff=[0]*signal.size
#print(ff)
w=2*np.pi/signal.size
t1=time.time()
for k in range(signal.size):
    for n in range(signal.size):
        ff[k]= ff[k] + signal[n]*np.exp(-1j*w*k*n) 
    ff[k]-ff[k]/signal.size
#print(ff)
t2=time.time()-t1
t3=time.time()
fft=np.fft.fft(signal)
t4=time.time()-t3
#print(fft)
print('DFS time',t2)
print('FFT time',t4)

ifft=ifft(fft)
print(ifft)
plt.plot(ifft,"*")
#plt.plot(xf, 2.0/N * np.abs(fft[0:N//2]))
plt.plot(np.abs(fft),"*")
plt.grid()
plt.show()