import numpy as np
from math import *
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
from numpy.polynomial.polynomial import Polynomial

def xrng(n):
    x = np.linspace(-1, 1, n + 1)
    return x 

# Q1 part 1 
# Runge Function
def rungeFunction(x):
    return 1 / ((25 * (x**2)) + 1)

y = []
for n in xrng(200):
    y.append(rungeFunction(n))

plt.plot(xrng(200), y)
plt.legend(['Runge Function'])
plt.title("Runge Function with n = 200")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()


## Q1 Part 2
# Lagrange interpolation Formula & plot
plt.plot(xrng(200), rungeFunction(xrng(200)))


poly = lagrange(xrng(6), rungeFunction(xrng(6)))
plt.plot(xrng(200), poly(xrng(200)), '--')


poly = lagrange(xrng(8), rungeFunction(xrng(8)))
plt.plot(xrng(200), poly(xrng(200)), ':')


poly = lagrange(xrng(10), rungeFunction(xrng(10)))
plt.plot(xrng(200), poly(xrng(200)), '-.')


plt.legend(['Runge Function', 'Lagrange n = 6', 'Lagrange n = 8', "Lagrange n = 10"])
plt.title("Lagrange Interpolation at Different 'n' Values")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()

## Q1 part 3
# Lagrange interpolation Formula & plot
# plotting runge and lagrangian first
def xrng(n):
    x = np.linspace(-1, 1, n + 1)
    return x 

plt.plot(xrng(200), rungeFunction(xrng(200)))

poly = lagrange(xrng(6), rungeFunction(xrng(6)))
plt.plot(xrng(200), poly(xrng(200)), '--')


poly = lagrange(xrng(6), rungeFunction(xrng(6)))
plt.plot(xrng(200), poly(xrng(200)), ':')


poly = lagrange(xrng(10), rungeFunction(xrng(10)))
plt.plot(xrng(200), poly(xrng(200)), '-.')


# Question 1 part 3
# Now plotting cubicspline on top of above plots
cspline = CubicSpline(xrng(10), rungeFunction(xrng(10)))
plt.plot(xrng(200), cspline(xrng(200)), ':')

plt.legend(['Runge Function', 'Lagrange n = 6', 'Lagrange n = 8', "Lagrange n = 10", 'Cubic Spline n = 10'])
plt.title("Cubic Spline On Top Of Lagrange Interpolation and Runge")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()

##From the above plot, we can observe the purple cubic spline 
# line follow the Runge function almost exactly while all the 
# Lagrange functions stray off the the function at the ends and 
# are not a good fit in the middle either. Thus we can conclude 
# that cubic spline is a better interpolation function than lagrange interpolation


# Q2 part 1
# Writing a Composite Trapezoidal integration function
def trapezoid(f_x, a, b, n):    
    h = (b-a)/float(n)
    integral = (f_x(a) + f_x(b))/2
    for i in range(1,n,1):
        a +=h
        integral += f_x(a)
        integral *= h;
        return integral

# Testing trapezoid function on g(t)
def g(t):    
    return exp(-t**2)

a = -2;  b = 2
n = 1000
result = trapezoid(g, a, b, n)
print(result)


# Q2 part 2
# Writing a Simpson Composite integration function
def simpson(f_x, a, b, n):    
    h = (b-a) / n
    integral=0.0 # initialize
    for i in range(1, int(n/2)): #even points        
        a += 2 * h
        integral += 4 * f_x(a)    
    
    for i in range(2, int(n/2)-1): #odd points        
        a += 2 * h
        integral += 2 * f_x(a)
    
    integral += f_x(a) + f_x(b)    
    integral *= h / 3    

    return integral

# Testing Simpson Composite integration on function(x)
def function(x): 
    return x

print(simpson(function, 0.0, 1.0, 100))

# Given our function S
def functionS(x):
    return exp(-((x-1)**2) / 2) / (2*pi)**0.5

# We get the following values for both our integration methods
print(f"Simpson Integration: {simpson(functionS, -100.0, 100.0, 100)}")
print(f"Trapezoid Integration: {trapezoid(functionS, -100.0, 100.0, 100)}")

# Since our trapezoidal integration gives a value of 0, we can conclude
# that simpson integration is better for this function.