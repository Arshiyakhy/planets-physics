import numpy as np
import matplotlib.pyplot as plt
from src.bodies import Body
from src.physics import acceleration, step_semi_implicit, step_explicit, energy, step_rk4, angular_momentum


sun = Body("Sun", mass=1.0, position=[0, 0], velocity=[0, 0])
dt = 0.025
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
# RK4
earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
rk4_positions = []
for i in range(steps):
    step_rk4(earth, sun.position, sun.mass, dt)
    rk4_positions.append(earth.position.copy())
rk4_positions = np.array(rk4_positions)
# Energy comparison
earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
symplectic_energy = []
for i in range(steps):
    step_semi_implicit(earth, sun.position, sun.mass, dt)
    symplectic_energy.append(energy(earth, sun.position, sun.mass))

earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
rk4_energy = []
for i in range(steps):
    step_rk4(earth, sun.position, sun.mass, dt)
    rk4_energy.append(energy(earth, sun.position, sun.mass))
# angular momentum comparison
earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
symplectic_angular_momentum = []
for i in range(steps):
    step_semi_implicit(earth, sun.position, sun.mass, dt)
    symplectic_angular_momentum.append(angular_momentum(earth))

earth = Body("Earth", mass=3e-6, position=[1.0, 0.0], velocity=[0.0, 2*np.pi])
rk4_angular_momentum = []
for i in range(steps):
    step_rk4(earth, sun.position, sun.mass, dt)
    rk4_angular_momentum.append(angular_momentum(earth))
# Angular momentum plot
plt.figure(figsize=(8, 5))
true_L = 2 * np.pi
plt.plot(symplectic_angular_momentum, label="Symplectic Euler")
plt.plot(rk4_angular_momentum, label="RK4")
plt.xlabel("Step")
plt.ylabel("Angular momentum")
plt.title("Angular momentum conservation: Symplectic vs RK4 (dt=0.025)")
plt.axhline(true_L, color='gray', linestyle='--',
            linewidth=0.8, label="True angular momentum")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("angular_momentum_comparison.png", dpi=150)
plt.show()
# Energy plot
plt.figure(figsize=(8, 5))
plt.plot(symplectic_energy, label="Symplectic Euler")
plt.plot(rk4_energy, label="RK4")
plt.axhline(-19.7402, color='gray', linestyle='--',
            linewidth=0.8, label="True energy")
plt.xlabel("Step")
plt.ylabel("Total energy")
plt.title("Energy conservation: Symplectic vs RK4 (dt=0.025)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("energy_comparison.png", dpi=150)
plt.show()
# Trajectory plot
plt.figure(figsize=(8, 8))
plt.plot(symplectic_positions[:, 0], symplectic_positions[:,
         1], label="Symplectic Euler", linewidth=1)
plt.plot(forward_positions[:, 0], forward_positions[:,
         1], label="Forward Euler", linewidth=1)
plt.plot(rk4_positions[:, 0], rk4_positions[:, 1], label="RK4", linewidth=1)
plt.plot(0, 0, 'yo', markersize=15, label="Sun")
plt.gca().set_aspect('equal')
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title("Earth's orbit: Symplectic vs Forward Euler vs RK4 (dt=0.025)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("orbit_comparison.png", dpi=150)
plt.show()
