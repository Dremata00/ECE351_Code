import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize' : 14})

#Task2
steps = 1e-3
t = np.arange(0, 10 + steps, steps)

print('Number of elements: len(t)= ', len(t), '\nFirst Element: t[0] = ', t[0]
      , '\nLast Element: t[len(t) - 1] = ', t[len(t) - 1])

def example1 (t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
            y[i] = np.cos(5*t[i]) + 2
    return y

y = example1(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) ')
plt.xlabel('seconds')
plt.title('y(t) = cos(t)')
