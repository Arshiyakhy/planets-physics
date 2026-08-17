from src.bodies import Body
from src.physics import acceleration

sun = Body("Sun", mass=1.0, position=[0, 0], velocity=[0, 0])
earth = Body("Earth", mass=3e-6, position=[1, 0], velocity=[0, 6.283])

a = acceleration(earth.position, sun.position, sun.mass)
print(a)
