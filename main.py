import numpy as np
import matplotlib.pyplot as plt
from src.bodies import Body
from src.physics import acceleration, hohmann_transfer, step_semi_implicit, step_explicit, energy, step_rk4, angular_momentum, step_semi_implicit_nbody, total_momentum, total_energy
from astroquery.jplhorizons import Horizons
from src.horizons import fetch_body

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

sun = Body("Sun", mass=1.0, position=[0, 0], velocity=[0, 0])
earth = Body("Earth", mass=3e-6, position=[1.0, 0], velocity=[0, 2*np.pi])
mars_r = 1.524
mars_v = np.sqrt(4*np.pi**2 / mars_r)
mars = Body("Mars", mass=3.2e-7, position=[mars_r, 0], velocity=[0, mars_v])

bodies = [sun, earth, mars]

nbody_steps = 3000
sun_positions, earth_positions, mars_positions = [], [], []

for i in range(nbody_steps):
    step_semi_implicit_nbody(bodies, dt)
    sun_positions.append(sun.position.copy())
    earth_positions.append(earth.position.copy())
    mars_positions.append(mars.position.copy())

sun_positions = np.array(sun_positions)
earth_positions = np.array(earth_positions)
mars_positions = np.array(mars_positions)

print("Sun final position:", sun.position)
print("Sun max distance from origin:", np.max(
    np.linalg.norm(sun_positions, axis=1)))
plt.figure(figsize=(8, 8))
plt.plot(earth_positions[:, 0], earth_positions[:, 1],
         label="Earth", linewidth=1)
plt.plot(mars_positions[:, 0], mars_positions[:, 1], label="Mars", linewidth=1)
plt.plot(sun_positions[:, 0], sun_positions[:, 1], label="Sun", linewidth=2)
plt.gca().set_aspect('equal')
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title("N-body simulation: Sun + Earth + Mars (75 years)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("nbody_orbits.png", dpi=150)
plt.show()

sun_positions, earth_positions, mars_positions = [], [], []
system_energy = []
system_momentum = []

for i in range(nbody_steps):
    step_semi_implicit_nbody(bodies, dt)
    sun_positions.append(sun.position.copy())
    earth_positions.append(earth.position.copy())
    mars_positions.append(mars.position.copy())
    system_energy.append(total_energy(bodies))
    system_momentum.append(np.linalg.norm(total_momentum(bodies)))

sun_positions = np.array(sun_positions)
earth_positions = np.array(earth_positions)
mars_positions = np.array(mars_positions)

plt.figure(figsize=(8, 5))
plt.plot(system_energy)
plt.axhline(system_energy[0], color='gray',
            linestyle='--', label="Initial energy")
plt.xlabel("Step")
plt.ylabel("Total system energy")
plt.title("N-body total energy conservation (Sun+Earth+Mars)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("nbody_energy.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(system_momentum)
plt.xlabel("Step")
plt.ylabel("|Total system momentum|")
plt.title("N-body total momentum conservation (Sun+Earth+Mars)")
plt.grid(True, alpha=0.3)
plt.savefig("nbody_momentum.png", dpi=150)
plt.show()


obj = Horizons(id='399', location='500@0',
               epochs={'start': '2026-01-01', 'stop': '2026-01-02', 'step': '1d'})
vectors = obj.vectors()
print(vectors)
print(vectors['x', 'y', 'z', 'vx', 'vy', 'vz'])

sun_real = fetch_body("Sun", "10", 1.0, "2026-01-01", "2026-01-02")
earth_real = fetch_body("Earth", "399", 3e-6, "2026-01-01", "2026-01-02")
mars_real = fetch_body("Mars", "499", 3.2e-7, "2026-01-01", "2026-01-02")

print(earth_real.position, earth_real.velocity)


sun_real = fetch_body("Sun", "10", 1.0, "2026-01-01", "2026-01-02")
earth_real = fetch_body("Earth", "399", 3e-6, "2026-01-01", "2026-01-02")
mars_real = fetch_body("Mars", "499", 3.2e-7, "2026-01-01", "2026-01-02")

real_bodies = [sun_real, earth_real, mars_real]
sim_steps = 200
sim_dt = 0.01

for i in range(sim_steps):
    step_semi_implicit_nbody(real_bodies, sim_dt)

print("Simulated Earth position after 2 years:", earth_real.position)
earth_actual_2028 = fetch_body(
    "Earth", "399", 3e-6, "2028-01-01", "2028-01-02")
print("Real Earth position on 2028-01-01:", earth_actual_2028.position)


# --- Jupiter hypothesis test ---
jupiter_real = fetch_body("Jupiter", "599", 9.5e-4, "2026-01-01", "2026-01-02")

# Rerun without Jupiter (reset to fresh 2026 state)
sun_real = fetch_body("Sun", "10", 1.0, "2026-01-01", "2026-01-02")
earth_real = fetch_body("Earth", "399", 3e-6, "2026-01-01", "2026-01-02")
mars_real = fetch_body("Mars", "499", 3.2e-7, "2026-01-01", "2026-01-02")
bodies_no_jupiter = [sun_real, earth_real, mars_real]

for i in range(sim_steps):
    step_semi_implicit_nbody(bodies_no_jupiter, sim_dt)

# Rerun WITH Jupiter (fresh state again)
sun_real2 = fetch_body("Sun", "10", 1.0, "2026-01-01", "2026-01-02")
earth_real2 = fetch_body("Earth", "399", 3e-6, "2026-01-01", "2026-01-02")
mars_real2 = fetch_body("Mars", "499", 3.2e-7, "2026-01-01", "2026-01-02")
jupiter_real2 = fetch_body("Jupiter", "599", 9.5e-4,
                           "2026-01-01", "2026-01-02")
bodies_with_jupiter = [sun_real2, earth_real2, mars_real2, jupiter_real2]

for i in range(sim_steps):
    step_semi_implicit_nbody(bodies_with_jupiter, sim_dt)

# Compare both against the real 2028 position
real_2028 = earth_actual_2028.position  # already fetched above

error_no_jupiter = np.linalg.norm(bodies_no_jupiter[1].position - real_2028)
error_with_jupiter = np.linalg.norm(
    bodies_with_jupiter[1].position - real_2028)

print("Error WITHOUT Jupiter:", error_no_jupiter, "AU")
print("Error WITH Jupiter:   ", error_with_jupiter, "AU")

dv1, dv2, dv_total = hohmann_transfer(1.0, 1.524)
print(f"Burn 1 (Earth departure): {dv1:.4f} AU/year")
print(f"Burn 2 (Mars arrival):    {dv2:.4f} AU/year")
print(f"Total delta-v:            {dv_total:.4f} AU/year")
