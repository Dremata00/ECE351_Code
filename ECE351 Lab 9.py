import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sig
#

step = 1e-2
t = np.arange(0, 2, step)

def fft(x,fs):
    N = len(x) # find the length of the signal
    X_fft = sig.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = sig.fftshift(X_fft) # shift zero frequency components
                                        # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
                                     # signal , (fs is the sampling frequency and
                                     # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    return freq, X_mag, X_phi

def fft2(x,fs):
    N = len(x) # find the length of the signal
    X_fft = sig.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = sig.fftshift(X_fft) # shift zero frequency components
                                        # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
                                     # signal , (fs is the sampling frequency and
                                     # needs to be defined previously in your code
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal
    for i in range(N):
        if X_mag[i] < 1e-10:
            X_phi[i] = 0
    return freq, X_mag, X_phi
    

# Task 1
y = np.cos(2*np.pi*t)
[freq, X_mag, X_phi] = fft(y,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.title('Task 1 : User Defined FFT of cos(2πt)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)

# Task 2
y = 5*np.sin(2*np.pi*t)
[freq, X_mag, X_phi] = fft(y,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.title('Task 2 : User Defined FFT of 5sin(2πt)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)

# Task 3
y =2*np.cos((2*np.pi*2*t) -2) + np.sin((2*np.pi*6*t)+3)**2
[freq, X_mag, X_phi] = fft(y,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.title('Task 3 : User Defined FFT of 2cos((2π*2t)-2) + sin^2((2π*6t)+3)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)

# Task 4
y = np.cos(2*np.pi*t)
[freq, X_mag, X_phi] = fft2(y,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.title('Task 4 : User Defined Refined FFT of cos(2πt)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)


y = 5*np.sin(2*np.pi*t)
[freq, X_mag, X_phi] = fft2(y,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.title('Task 4 : User Defined Refined FFT of 5sin(2πt)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)


y =2*np.cos((2*np.pi*2*t) -2) + np.sin((2*np.pi*6*t)+3)**2
[freq, X_mag, X_phi] = fft2(y,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.title('Task 3 : User Defined Refined FFT of 2cos((2π*2t)-2) + sin^2((2π*6t)+3)')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)

# Task 5
def x(t,N):
    x = 0
    for k in range(1,N):
        x += 2/(k*np.pi)*(1-np.cos(k*np.pi))*np.sin(k*2*np.pi*t/8)
    return x

t = np.arange(0, 16, step)
f = x(t,15)
[freq, X_mag, X_phi] = fft2(f,100)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, f)
plt.grid()
plt.title('Task 5 : User Defined FFT of Fourier series approximation of Square Function')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(freq , X_mag) 
plt.ylabel('|X(f)|')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(freq , X_mag) 
plt.xlim(-2,2)
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(freq , X_phi) 
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.stem(freq , X_phi) 
plt.xlabel('f [Hz]')
plt.xlim(-2,2)
plt.grid(True)