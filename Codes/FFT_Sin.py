import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft,ifft


t=np.array([i for i in range(6000)])
S1=np.sin(t/50)
S2=np.sin(t/100)
S3=np.sin(t/130)

plt.figure("1")
plt.subplot(1, 3, 1) # row 1, col 2 index 1
plt.plot(S1,"-")
plt.grid()
plt.subplot(1, 3, 2)
plt.plot(S2,"-")
plt.grid()
plt.subplot(1, 3, 3)
plt.plot(S3,"-")
plt.grid()


plt.figure("2")
S4=3*S1+2*S2+S3
plt.plot(S4,"-")
plt.grid()


ff=[0]*S4.size
#print(ff)
w=2*np.pi/S4.size

#print(ff)

fft=np.fft.fft(S4)
#print(fft)


ifft=ifft(fft)
print(ifft)
#plt.xlim(0,100)
#plt.plot(ifft,"-")

#plt.plot(xf, 2.0/N * np.abs(fft[0:N//2]))
plt.figure("3")
plt.plot(np.abs(fft),"-")
plt.grid()


plt.show()