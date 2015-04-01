"""

This program is a simplified version of a program "sample-plots.py" by 
Bruce Cohen 3/5/12.
Modified 3/17/15 by ds

"""

import matplotlib.pyplot as plt

from math import *
from numpy.linalg import solve

def lagrange(points):

def polynom(x):
plt.title(r'$5^{th}$ degree Maclaurin polynomial vs $\sin(x)$')
def taylor_sine(x):
    return(x-(x**3)/(2*3) + (x**5)/(2*3*4*5))
def plotGraph(fx, points):
    x = range(-5, 5)
    y = map(fx, x)
    print y
    plt.plot(x, y,linewidth=0.5)
    xList = []
    yList = []
    for xp, yp in points:
	xList.append(xp)
	yList.append(yp)
    print xList 
    print yList
    plt.plot(xList, yList, 'ro')
    plt.show()
def lagrange(points):
    Px = 0
    n = len(points)
    for i in range(n):
	xi,yi = points[i]
	totalMul = 1
	for j in range(n):
	    if i ==j:
		continue
	    xj, yj = points[j]
	    totalMul *= (x-xj)/float(xi - xj)
	Px += yi*totalMul
   return Px
def q3():
    "Lagrange & Hermite Interpolating Polynomials"
    points[(0,1/6),(1/6,1/3),(1/3,1/2),(1/2,7/12),(7/12,2/3),(2/3,3/4),(3/4,5/6),(5/6,11/12),(11/12,1)]
    fx = 1.6e**-2x*sin(3pix)
    Px = lagrange(points)
    plot(Px, points):


taylor_label = r'$x- \frac{x^3}{3!}+\frac{x^5}{5!}$'

n = 36 # number of intervals per unit
delta_theta = pi/n

# Compute lists of x and y coordinates for the two graphs
thetas = [-pi+k*delta_theta for k in range(3*n+1)]
taylor_ys = [taylor_sine(t) for t in thetas]
sin_ys = [sin(t) for t in thetas]

# Plot the curves and (border) axes
plt.plot(thetas, taylor_ys, color='red', label=taylor_label)
plt.plot(thetas, sin_ys, color='blue', label= r'$\sin(x)$')
plt.axis([-pi, 2*pi , -2, 2], 3/4 )

# draw axes centered at the origin
# first a default hline at y=0 that spans the xrange
plt.axhline(y=0, color='black')

# second a default vline at x=0 that spans the yrange
plt.axvline(x=0, color='black')

plt.legend(loc='lower right')

# Display graphs
plt.show()

# This doesn't seem to be working.
plt.savefig("taylor_sin.png")
if __name__ == "__main__":
   q3()

