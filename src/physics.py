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


def derivatives(pos, vel, other_pos, other_mass):
    acc = acceleration(pos, other_pos, other_mass)
    return vel, acc   # (dr/dt, dv/dt)


def step_rk4(body, other_pos, other_mass, dt):
    r1, v1 = body.position, body.velocity
    k1r, k1v = derivatives(r1, v1, other_pos, other_mass)

    r2 = r1 + dt/2 * k1r
    v2 = v1 + dt/2 * k1v
    k2r, k2v = derivatives(r2, v2, other_pos, other_mass)

    r3 = r1 + dt/2 * k2r
    v3 = v1 + dt/2 * k2v
    k3r, k3v = derivatives(r3, v3, other_pos, other_mass)

    r4 = r1 + dt * k3r
    v4 = v1 + dt * k3v
    k4r, k4v = derivatives(r4, v4, other_pos, other_mass)

    body.position = r1 + dt/6 * (k1r + 2*k2r + 2*k3r + k4r)
    body.velocity = v1 + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
