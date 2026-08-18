import numpy as np

G = 4 * np.pi**2


def acceleration(pos, other_pos, other_mass):
    r_vec = pos - other_pos
    r_mag = np.linalg.norm(r_vec)
    return -G * other_mass * r_vec / r_mag**3


def step_semi_implicit(body, other_pos, other_mass, dt):
    a = acceleration(body.position, other_pos, other_mass)
    body.velocity += a * dt
    body.position += body.velocity * dt


def step_explicit(body, other_pos, other_mass, dt):
    a = acceleration(body.position, other_pos, other_mass)
    old_velocity = body.velocity.copy()
    body.velocity = body.velocity + a * dt
    body.position = body.position + old_velocity * dt


def energy(body, other_pos, other_mass, G=4*np.pi**2):
    r = np.linalg.norm(body.position - other_pos)
    v = np.linalg.norm(body.velocity)
    kinetic = 0.5 * v**2          # per unit mass of the orbiting body
    potential = -G * other_mass / r
    return kinetic + potential
