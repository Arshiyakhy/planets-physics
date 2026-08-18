import numpy as np
import matplotlib.pyplot as plt
from src.bodies import Body
from src.physics import acceleration, step_semi_implicit, step_explicit, energy


sun = Body("Sun", mass=1.0, position=[0, 0], velocity=[0, 0])
dt = 0.05
steps = 1000

# semi-implicit (symplectic) Euler
earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
symplectic_positions = []
for i in range(steps):
    step_semi_implicit(earth, sun.position, sun.mass, dt)
    symplectic_positions.append(earth.position.copy())
symplectic_positions = np.array(symplectic_positions)

# explicit (forward) Euler
earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
forward_positions = []
for i in range(steps):
    step_explicit(earth, sun.position, sun.mass, dt)
    forward_positions.append(earth.position.copy())
forward_positions = np.array(forward_positions)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(symplectic_positions[:, 0], symplectic_positions[:,
         1], label="Symplectic Euler", linewidth=1)
plt.plot(forward_positions[:, 0], forward_positions[:,
         1], label="Forward Euler", linewidth=1)
plt.plot(0, 0, 'yo', markersize=15, label="Sun")  # sun at origin
plt.gca().set_aspect('equal')  # so orbits aren't visually distorted
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title("Earth's orbit: Symplectic vs Forward Euler (dt=0.05)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("orbit_comparison.png", dpi=150)
plt.show()
