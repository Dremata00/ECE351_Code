"""
    Mata, Dre
    V00556648
    Section 51
    EE
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update ({'font.size': 18})

#Problem one

print (np.arange(20, -15, -1))
print(np.linspace(0+0.j, 5.+5.j, 6))
print(np.array([0 ,-9, -12, -15, -18, -21, -24, -27, -30, -33, -36]))

#Problem two

def x1(t):
    return np.sin(5*np.pi*t)

def x2(t):
    return 4*np.cos(0.6*np.pi*t)

def x3(t):
    return 5*np.e**(-0.2*t)*np.sin(3.2*np.pi*t)

def x4(t):
    return np.e**(-2*t) + 2*np.e**(-4.6*t) + 3*np.e**(-0.1*t)

step = 1e-3
t = np.arange(0, 6, step)

plt.figure(figsize = (16,16))
plt.suptitle('ECE 351 Final - Question 2')


plt.subplot(2, 2, 1)
plt.plot(t, x1(t), 'k', label = 'sin(5*pi*t)')
plt.ylim(-2,2)
plt.xlim(0,6)
plt.locator_params(axis = "both", integer = True, tight = True)
plt.title('X1')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend(loc ='upper right')

plt.subplot(2, 2, 2)
plt.plot(t, x2(t), 'b', label = '4cos(0.6*pi*t)')
plt.ylim(-5,5)
plt.xlim(0,6)
plt.locator_params(axis = "both", integer = True, tight = True)
plt.title('X2')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend(loc ='upper right')

plt.subplot(2, 2, 3)
plt.plot(t, x3(t), 'm', label = '5e^-(0.2)t sin(3.2*pi*t')
plt.ylim(-5,5)
plt.xlim(0,6)
plt.locator_params(axis = "both", integer = True, tight = True)
plt.title('X3')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend(loc = 'upper right')

plt.subplot(2, 2, 4)
plt.plot(t, x4(t), 'g', label = 'e^-(2t)+2e^-(4.6t)+3e^-(0.1t)')
plt.ylim(2,6)
plt.xlim(0,6)
plt.locator_params(axis = "both", integer = True, tight = True)
plt.title('X4')
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend(loc = 'upper right')

# Problem 3

num = [3, 2, 26]
den = [1, 7, -11]

[z, p, _] = sig.tf2zpk(num, den)
print('Zeros:', z)
print('Poles:', p)

t_out, y_out = sig.impulse((num,den), T = t)

plt.figure(figsize = (13, 13))
plt.subplot(2, 1, 1)
plt.plot(t_out, y_out, 'c', label = 'Impluse ')
plt.xlim(0,10)
plt.grid()
plt.title('Problem 3')
plt.xlabel('t')
plt.ylabel('y(t)')

t_out, y_out = sig.step((num,den), T = t)

plt.subplot(2, 1, 1)
plt.plot(t_out,y_out,  'r', label = 'Step')
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend(loc='upper left')

'''
Question 3: Both the inpulse and step response are unstable
'''

# Problem 4
step = 1e-1
t = np.arange(0, 6, step)

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

tconv = np.arange(0, 2*t[len(t)-1]+step, step)


plt.figure(figsize = (20, 16))
plt.suptitle('ECE 351 Final - Question 4')

f1 = conv(x2(t), x3(t))
f2 = conv(x2(t), x4(t))
f3 = conv(x4(t), x1(t))
f4 = conv(x1(t), x2(t))

plt.subplot(2, 2, 1)
plt.plot(tconv, f1/10,'k', label = 'X2*X3')
plt.title('f1')
plt.xlabel('t [s]')
plt.ylabel('f(t)')
plt.ylim(-4,4)
plt.xlim(0, 12)
plt.grid(True)
plt.legend(loc = 'upper right')

plt.subplot(2, 2, 2)
plt.plot(tconv, f2/10,'b', label = 'X2*X4')
plt.locator_params(axis = "both", integer = True, tight = True)
plt.title('f2')
plt.xlabel('t [s]')
plt.ylabel('f(t)')
plt.grid(True)
plt.legend(loc = 'upper right')


plt.subplot(2, 2, 3)
plt.plot(tconv, f3/10,'m', label = 'X4*X1')
plt.title('f3')
plt.xlabel('t [s]')
plt.ylabel('f(t)')
plt.grid(True)
plt.legend(loc ='upper right')

plt.subplot(2, 2, 4)
plt.plot(tconv, f4/10,'m', label = 'X1*X2')
plt.title('f4')
plt.xlabel('t [s]')
plt.ylabel('f(t)')
plt.grid(True)
plt.legend(loc = 'lower right')
