import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# Part one task 1

R = 1000
L = 0.027
C = 100 * (10 ** -9)

step = 1*10**(-8)
t = np.arange(0, 0.0012+step, step)

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def y(t,R,L,C):
    alpha = -1/(2 * R * C)
    
    omega = 1/2 * np.sqrt((1/(R * C)) ** 2 - 4 * (1/np.sqrt(L * C)) ** 2 + 0 * 1j)
    
    p = alpha + omega
    
    g = 1/(R*C) * p
    
    gMag = np.abs(g)
    
    gAngle = np.angle(g)
    
    return gMag/np.abs(omega)*np.e**(alpha*t)*np.sin(np.abs(omega)*t+gAngle)*u(t)

y = y(t,R,L,C)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.grid()
plt.title('Hand Calculated Plot')
plt.xlabel('t')
plt.ylabel('y(t)')

# Part one task 2

numerator = [0,1/(R * C),0]
denominator = [1,1/(R * C),1/(L * C)]

tout, yout = sig.impulse((numerator,denominator), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.title('Plot using scripy.signal')
plt.xlabel('t')
plt.ylabel('y(t)')

#Part 2

tout, yout = sig.step((numerator,denominator), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.title('Step Response of H(s)')
plt.xlabel('t')
plt.ylabel('y(t)')
