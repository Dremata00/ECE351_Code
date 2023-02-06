import numpy as np
import matplotlib.pyplot as plt

# Part one Task 1
steps = 1e-1
t = np.arange(-5, 10 + steps, steps)

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
    
def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y


def f1(t):
    return u(t - 2) - u(t - 9)

def f2(t):
    return np.exp(-t) * u(t)

def f3(t):
    return r(t - 2) * (u(t - 2) - u(t - 3)) + r(4 - t) * (u(t - 3) - u(t - 4))

# Part one Task 2

y1 = f1(t)
y2 = f2(t)
y3 = f3(t)


plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t[range(len(y1))], y1)
plt.grid()
plt.ylabel('y(t) ')
plt.xlabel('seconds')
plt.title('Plot for Part 1 Task 2')

plt.subplot(2, 1, 1)
plt.plot(t[range(len(y2))], y2)
plt.grid()

plt.subplot(2, 1, 1)
plt.plot(t[range(len(y3))], y3)
plt.grid()

# Part Two Task 1

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

# Part two Task 2

Nt = len(t)

tApp = np.arange(0, 3 * t[Nt - 1] + steps,steps)

cF1F2 = conv(f1(t),f2(t))

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(tApp, cF1F2)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Convalution of f1 and f2')

# Part two Task 3

cF2F3 = conv(f2(t),f3(t))

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tApp, cF2F3)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Convalution of f2 and f3')

# Part two Task 4

cF1F3 = conv(f1(t),f3(t))

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tApp, cF1F3)
plt.grid()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Convalution of f1 and f3')
