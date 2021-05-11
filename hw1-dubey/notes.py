import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import NaN

# def Question1and2(Mstar, Mplanet, d):
x = np.linspace(-2,2,84)
y = np.linspace(-2,2,84)
t = np.linspace(-2,2,84)
objposX, objposY = np.meshgrid(x,y)

G = -1
Mplanet = 1
Mstar = 100*Mplanet
M3  = 1
d = 1

starlocX = - d / 2
planetlocX = d / 2
Yloc, planetlocY, starlocY = 0, 0, 0

r2planetX = objposX - planetlocX
r2planetY = objposY - planetlocY
r2planet = r2planetX**2 + r2planetY**2

r2starX = objposX - starlocX
r2starY = objposY - starlocY
r2star = r2starX**2 + r2starY**2

gplanet = G*Mplanet/r2planet
gstar = G*Mstar/r2star
Z = M3 * (gplanet * np.sqrt(r2planet) + gstar * np.sqrt(r2star))
L = M3 * (gstar * np.sqrt(r2star) + gplanet * np.sqrt(r2planet)) * t
if Mstar/Mplanet >= 70:
    remove_radiiP = 0.01
    remove_radiiS = 0.1
else:
    remove_radiiP = 0.1
    remove_radiiS = 0.15

gplanet[r2planet < remove_radiiP] = NaN
gstar[r2star < remove_radiiS] = NaN


Pcostheta = r2planetX / r2planet
Psintheta = r2planetY / r2planet

Scostheta = r2starX / r2star
Ssintheta = r2starY / r2star

gX = (gplanet*Pcostheta + gstar*Scostheta)
gY = (gplanet*Psintheta + gstar*Ssintheta)

fig, ax = plt.subplots(figsize = (7,7))

ax.quiver(objposX, objposY, gX, gY)
ax.contour(x, y, Z)
ax.add_patch(plt.Circle((planetlocX, planetlocY), 0.01, color='blue'))
ax.add_patch(plt.Circle((starlocX, starlocY), 0.02, color='yellow'))


plt.show()