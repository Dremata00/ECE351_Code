import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize' : 14})

#Task 1
steps = 1e-2
t = np.arange(-10, 17 + steps, steps)

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

def func2(t):
    return r(-t) - r(-t - 3) + 5 * u(-t - 3) - 2 * u(-t - 6) - 2 * r(-t - 6)

y1 = func2(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t[range(len(y1))], y1)
plt.grid()
plt.ylabel('y(t) ')
plt.xlabel('seconds')
plt.title('Part 3 task 1')
print ('formula: ', len(t)/3)

#Task 2
def func3(t):
    return r(t - 4) - r(t - 7) + 5 * u(t - 7) - 2 * u(t - 10) - 2 * r(t - 10)
    
y2 = func3(t)

#plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t[range(len(y1))], y2)
plt.grid()
plt.ylabel('y(t) ')
#plt.xlabel('seconds')
plt.title('Part 3 task 2')
print ('formula: ', len(t)/3)

#Task 2.2
def func4(t):
    return r(-t - 4) - r(-t - 7) + 5 * u(-t - 7) - 2 * u(-t - 10) - 2 * r(-t - 10)
    
y3 = func4(t)

#plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t[range(len(y1))], y3)
plt.grid()
plt.ylabel('y(t) ')
#plt.xlabel('seconds')