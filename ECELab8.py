import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def ak(k):
    return 0

def bk(k):
    return 2/(k*np.pi)*(1-np.cos(k*np.pi))

print("a0 = ", ak(0), "  a1 = ", ak(1), "  b1 = ", bk(1),
      "  b2 = ", bk(2), "  b3 = ", bk(3),)


step = 1e-2
t = np.arange(0, 20, step)

def x(t,N):
    x = 0
    for k in range(1,N):
        x += 2/(k*np.pi)*(1-np.cos(k*np.pi))*np.sin(k*2*np.pi*t/8)
    return x

k = 1
y = 2/(k*np.pi)*(1-np.cos(k*np.pi))*np.sin(k*2*np.pi*t/8)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title(1)

N = [3,15,50,150,1500]

for i in N:
    f = x(t,i)

    plt.figure(figsize = (10, 7))
    plt.subplot(3, 1, 1)
    plt.plot(t, f)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title(i)

