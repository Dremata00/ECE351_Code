import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

step = 1000
w = np.arange(10**3, 10**6+step, step)

R = 1e3
L = 27e-3
C = 100e-9

# Part 1
def Hmag(w):
    RC = R*C
    LC = L*C
    w2 = w**2
    return 20*np.log10((w/RC)/(np.sqrt((-w2+1/LC)**2+(w/RC)**2)))

def Hang(w):
    return (np.pi/2-np.arctan((w/(R*C))/(-w**2+1/(L*C))))*180/np.pi

def adj(Hang):
    for i in range(len(Hang)):
        if Hang[i] > 90:
            Hang[i] = Hang[i] - 180
    return Hang

def HangHz(w):
    return (np.pi/2-np.arctan((w/(R*C))/(-w**2+1/(L*C))))

def adjHz(HangHz):
    for i in range(len(Hang)):
        if HangHz[i] > 90:
            HangHz[i] = Hang[i] - 180
    return HangHz

Hmag = Hmag(w)
adjustHang = adj(Hang(w))


plt.figure(figsize = (12, 8))

plt.subplot(2, 1, 1)
plt.semilogx(w,Hmag)
plt.title('')
plt.xlabel('rad/s')
plt.ylabel('Magnitude(dB)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(w,adjustHang)
plt.title('')
plt.xlabel('rad/s')
plt.ylabel('Angle')
plt.grid(True)

# Part 1 Task 2
num = [1/(R*C),0]
den = [1,1/(R*C),1/(L*C)]


w, mag, phase = sig.bode((num,den),w)

plt.figure(figsize = (12, 8))

plt.subplot(2, 1, 1)
plt.semilogx(w, mag)    # Bode magnitude plot
plt.xlabel('rad/s')
plt.ylabel('Magnitude(dB)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)  # Bode phase plot
plt.xlabel('rad/s')
plt.ylabel('Angle')
plt.grid(True)

# Part 2
def x(t):
    return np.cos(2*np.pi*100*t)+np.cos(2*np.pi*3024*t)+np.sin(2*np.pi*50000*t)

fs = 1e5
t = np.arange(0, 0.01+1/fs, 1/fs)

numZ, denZ = sig.bilinear(num, den,fs)
y = sig.lfilter(numZ,denZ,x(t))

plt.figure(figsize = (16, 12))
plt.subplot(2, 1, 1)
plt.plot(t, x(t))
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title("input signal")

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("output signal")
