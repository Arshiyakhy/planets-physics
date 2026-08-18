from src.bodies import Body
from src.physics import acceleration, step

sun = Body("Sun", mass=1.0, position=[0, 0], velocity=[0, 0])
earth = Body("Earth", mass=3e-6, position=[1, 0], velocity=[0, 6.283])

a = acceleration(earth.position, sun.position, sun.mass)
print(a)
for _ in range(1000):
    step(earth, sun.position, sun.mass, 0.001)

print(earth.position, earth.velocity)
