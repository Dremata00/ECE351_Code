import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize' : 14})
#Task3
steps = 1e-2
t = np.arange(-5, 10 + steps, steps)

print('Number of elements: len(t)= ', len(t), '\nFirst Element: t[0] = ', t[0]
      , '\nLast Element: t[len(t) - 1] = ', t[len(t) - 1])

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

def func1(t):
    return r(t) - r(t - 3) + 5 * u(t - 3) - 2 * u(t - 6) - 2 * r(t - 6)

y = func1(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t[range(len(y))], y)
plt.grid()
plt.ylabel('y(t) ')
#plt.xlabel('seconds')
plt.title('Plot for Lab 2')
print ('formula: ', len(t)/3)

#Task2
#step
def func2(t):
    return u(t)

y2 = func2(t)

#plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t[range(len(y2))], y2)
plt.grid()
plt.ylabel('y(t) ')
plt.xlabel('seconds')
plt.title('Step and Ramp Functions')
print ('formula: ', len(t)/3)

#ramp
def func3(t):
    return r(t)

y3 = func3(t)

#plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t[range(len(y3))], y3)
plt.grid()
plt.ylabel('y(t) ')
#plt.xlabel('seconds')
#plt.title('Plot for Lab 2')
print ('formula: ', len(t)/3)