import numpy as np
import matplotlib.pyplot as plt

step = 0.1
t = np.arange(-10, 10 + step, step)

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

# Part one Task one

def f1(t):
    return (np.e ** (-2 * t)) * (u(t) - u(t - 3))

def f2(t):
    return u(t - 2) - u(t - 6)

def f3(t):
    return np.cos(2 * np.pi * 0.25 * t) * u(t)

plt.figure(figsize = (12, 8))

plt.subplot(3, 1, 1)
plt.plot(t,f1(t))
plt.title('h1(t),h2(t),h3(t)')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t,f2(t))
plt.ylabel('f(t)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t,f3(t))
plt.ylabel('f(t)')
plt.grid(True)

plt.show()

# Part one Task two

def conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1App = np.append(f1,np.zeros((1,Nf2 - 1)))
    f2App = np.append(f2,np.zeros((1,Nf1 - 1)))
    result = np.zeros(f1App.shape)
    for i in range (Nf1 + Nf2 - 2):
        result[i] = 0
        
        for j in range (Nf1):
            if ((i - j) + 1 > 0):
                try:
                    result[i] += f1App[j] * f2App[i - j + 1]
                except:
                    print(i - j)
    return result

Nt = len(t)
tApp = np.arange(-20, 2 * t[Nt - 1] + step,step)

convf1 = conv(f1(t),u(t))

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tApp,convf1)
plt.grid()
plt.title('Step response of h1(t)')
plt.xlabel('t')
plt.ylabel('h(t)')

convf2 = conv(f2(t),u(t))

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tApp,convf2)
plt.grid()
plt.title('Step response of h2(t)')
plt.xlabel('t')
plt.ylabel('h(t)')

convf3 = conv(f3(t),u(t))

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tApp,convf3)
plt.grid()
plt.title('Step response of h3(t)')
plt.xlabel('t')
plt.ylabel('h(t)')
