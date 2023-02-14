import numpy as np
import matplotlib.pyplot as plt

step = 0.1
t = np.arange(-20, 20 + step, step)


def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def y1(t):
    return 0.5 * (1 - (np.exp(-2 * t))) * u(t) - 0.5 * (1 - np.exp(-2 * (t - 3))) * u(t - 3)

def y2(t):
    return (t - 2) * u(t - 2) - (t - 6) * u(t - 6)

def y3(t):
    return 1/(2 * np.pi * 0.25) * np.sin((2 * np.pi * 0.25) * t) * u(t)

tApp2 = np.arange(-20, 20 + step, step)


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,y1(t))
plt.title('Hand Caluculated convolutions')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t,y2(t))
plt.ylabel('f(t)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t,y3(t))
plt.ylabel('f(t)')
plt.grid(True)

plt.show()