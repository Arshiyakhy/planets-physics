from src.bodies import Body
from src.physics import acceleration, step, energy, step_symplectic
import numpy as np

sun = Body("Sun", mass=1.0, position=[0, 0], velocity=[0, 0])
earth = Body("Earth", mass=3e-6, position=[1, 0], velocity=[0, 6.283])

a = acceleration(earth.position, sun.position, sun.mass)
print(a)
earth.position = np.array([1.0, 0.0])
earth.velocity = np.array([0.0, 2*np.pi])

for i in range(1000):
    step(earth, sun.position, sun.mass, 0.05)
    if i % 20 == 0:
        print(energy(earth, sun.position, sun.mass))
earth = Body("Earth", 3e-6, [1.0, 0.0], [0.0, 2*np.pi])
for i in range(1000):
    step_symplectic(earth, sun.position, sun.mass, 0.05)
    if i % 20 == 0:
        print(energy(earth, sun.position, sun.mass))
print(earth.position, earth.velocity)
