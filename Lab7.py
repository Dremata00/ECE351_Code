import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# Part 1 Task 1

def G(s):
    return (s + 9)/((s+2)*(s+4)*(s-8))

def A(s):
    return (s + 4)/((s+3)*(s+1))

def B(s):
    return (s + 12) * (s + 14)

print('Hand Written')
print('G(s)')
print('zero: [-9]', 'poles: [8. -4. -2]')

print('B(s)')
print('zero: [-4]', 'poles: [-3. -1]')

print('A(s)')
print('zero: [-14. -12]')

# Part 1 Task 2

Gnum = [1,9]
Gden = [1,-2,-40,-64]

[Gz, Gp, _] = sig.tf2zpk(Gnum, Gden)
print('Using .tf2zpk()')
print('G(s)')
print('zero: ', Gz, 'poles: ', Gp)


Anum = [1,4]
Aden = [1,4,3]

[Az, Ap, _] = sig.tf2zpk(Anum, Aden)
print('A(s)')
print('zero: ', Az, 'poles: ', Ap)

Bnum = [1,26,168]

Bz=np.roots(Bnum)
print('B(s)')
print('zero: ', Bz)

# part 1 Task 5

step = 1e-2
t = np.arange(0, 7+step, step)

Dnum = sig.convolve(Anum, Gnum)
Dden = sig.convolve(Bnum, Gden)
Open = Dden

tout, yout = sig.step((1,Open), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.title('Open-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('y(t)')

# Part 2

Cnum = Dnum
Cden = (1 + sig.convolve(Gnum, Bnum))


[Cz,Cp, _] = sig.tf2zpk(Cnum, Cden)
print('part 2')
print('Y(s)')
print(Cz,Cp)

tout, yout = sig.step((Cnum,Cden), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.title('Closed-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('y(t)')

