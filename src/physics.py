import numpy as np

G = 4 * np.pi**2


def acceleration(pos, other_pos, other_mass):
    r_vec = pos - other_pos
    r_mag = np.linalg.norm(r_vec)
    return -G * other_mass * r_vec / r_mag**3


def step(body, other_pos, other_mass, dt):
    a = acceleration(body.position, other_pos, other_mass)
    body.velocity += a * dt
    body.position += body.velocity * dt
