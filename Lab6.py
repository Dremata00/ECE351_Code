import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

step = 1 * 10**(-8)
t = np.arange(0, 0.0012 +step, step)

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

# Part 1

def y(t):
    return 1/4 * (18 * np.e**(-6 * t) - 14 * np.e**(-4 * t)) * u(t)

y = y(t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.grid()
plt.title('Hand Calculated Plot')
plt.xlabel('t')
plt.ylabel('y(t)')

num = [1,6,12]
den = [1,10,24]

tout, yout = sig.step((num,den), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.title('Plot using scipy.signal.step()')
plt.xlabel('t')
plt.ylabel('y(t)')

[R1, P1, _] = sig.residue(num , den)

print("Roots: ")
print (R1)
print("Poles: ")
print(P1)

# Part 2

num = [25250]
den = [1,18,218,2036,9085,25250, 0]
[R2, P2, _] = sig.residue(num , den)

print("Roots: ")
print (R2)
print("Poles: ")
print(P2)

step = 1e-2
t = np.arange(0, 4.5 + step, step)


def yc(t):
    y = 0
    
    for i in range(len(P2)):
        alpha = np.real(P2[i])
        omega = np.imag(P2[i])
        kmag = np.abs(R2[i])
        kangle = np.angle(R2[i])
        y += kmag*np.e**(alpha*t)*np.cos(omega*t+kangle)*u(t)
        
    return y

y = yc(t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,y)
plt.grid()
plt.title('Plot using cosin method')
plt.xlabel('t')
plt.ylabel('y(t)')

den = [1,18,218,2036,9085,25250]

tout, yout = sig.step((num,den), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.title('Plot using scripy.signal.step')
plt.xlabel('t')
plt.ylabel('y(t)')


