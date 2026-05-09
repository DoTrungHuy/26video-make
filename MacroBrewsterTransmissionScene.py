from manim import *
import numpy as np
import math
import random
import cmath
from typing import List, Tuple, Dict, Optional, Union, Callable


class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def mag(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def mag_sq(self) -> float:
        return self.x ** 2 + self.y ** 2

    def norm(self) -> 'Vector2D':
        m = self.mag()
        if m < 1e-12:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / m, self.y / m)

    def add(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def sub(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def mul(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

    def dot(self, other: 'Vector2D') -> float:
        return self.x * other.x + self.y * other.y


class Vector3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Vector3D':
        return Vector3D(float(arr[0]), float(arr[1]), float(arr[2]))

    def clone(self) -> 'Vector3D':
        return Vector3D(self.x, self.y, self.z)

    def add(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def mul(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def div(self, scalar: float) -> 'Vector3D':
        if abs(scalar) < 1e-12:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def mag_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def mag(self) -> float:
        return math.sqrt(self.mag_sq())

    def norm(self) -> 'Vector3D':
        m = self.mag()
        if m < 1e-12:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / m, self.y / m, self.z / m)

    def rotate_x(self, angle: float) -> 'Vector3D':
        c = math.cos(angle)
        s = math.sin(angle)
        ny = self.y * c - self.z * s
        nz = self.y * s + self.z * c
        return Vector3D(self.x, ny, nz)

    def rotate_y(self, angle: float) -> 'Vector3D':
        c = math.cos(angle)
        s = math.sin(angle)
        nx = self.x * c + self.z * s
        nz = -self.x * s + self.z * c
        return Vector3D(nx, self.y, nz)

    def rotate_z(self, angle: float) -> 'Vector3D':
        c = math.cos(angle)
        s = math.sin(angle)
        nx = self.x * c - self.y * s
        ny = self.x * s + self.y * c
        return Vector3D(nx, ny, self.z)

    def distance_to(self, other: 'Vector3D') -> float:
        return self.sub(other).mag()

    def lerp(self, other: 'Vector3D', t: float) -> 'Vector3D':
        return self.add(other.sub(self).mul(t))

    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        o_norm = other.norm()
        return o_norm.mul(self.dot(o_norm))

    def reflect_plane(self, normal: 'Vector3D') -> 'Vector3D':
        n = normal.norm()
        return self.sub(n.mul(2.0 * self.dot(n)))


class Matrix3x3:
    def __init__(self):
        self.m = [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]

    def determinant(self) -> float:
        return (self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1]) -
                self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0]) +
                self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]))

    def transpose(self) -> 'Matrix3x3':
        res = Matrix3x3()
        for i in range(3):
            for j in range(3):
                res.m[i][j] = self.m[j][i]
        return res


class Matrix4x4:
    def __init__(self):
        self.m = [[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]]

    def multiply(self, other: 'Matrix4x4') -> 'Matrix4x4':
        res = Matrix4x4()
        for i in range(4):
            for j in range(4):
                val = 0.0
                for k in range(4):
                    val += self.m[i][k] * other.m[k][j]
                res.m[i][j] = val
        return res

    def translate(self, tx: float, ty: float, tz: float) -> 'Matrix4x4':
        t_mat = Matrix4x4()
        t_mat.m[0][3] = tx
        t_mat.m[1][3] = ty
        t_mat.m[2][3] = tz
        return self.multiply(t_mat)

    def scale(self, sx: float, sy: float, sz: float) -> 'Matrix4x4':
        s_mat = Matrix4x4()
        s_mat.m[0][0] = sx
        s_mat.m[1][1] = sy
        s_mat.m[2][2] = sz
        return self.multiply(s_mat)

    def rotate_x(self, angle: float) -> 'Matrix4x4':
        r_mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        r_mat.m[1][1] = c
        r_mat.m[1][2] = -s
        r_mat.m[2][1] = s
        r_mat.m[2][2] = c
        return self.multiply(r_mat)

    def rotate_y(self, angle: float) -> 'Matrix4x4':
        r_mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        r_mat.m[0][0] = c
        r_mat.m[0][2] = s
        r_mat.m[2][0] = -s
        r_mat.m[2][2] = c
        return self.multiply(r_mat)

    def rotate_z(self, angle: float) -> 'Matrix4x4':
        r_mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        r_mat.m[0][0] = c
        r_mat.m[0][1] = -s
        r_mat.m[1][0] = s
        r_mat.m[1][1] = c
        return self.multiply(r_mat)


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def multiply(self, q: 'Quaternion') -> 'Quaternion':
        w = self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z
        x = self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y
        y = self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x
        z = self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        return Quaternion(w, x, y, z)

    def normalize(self) -> 'Quaternion':
        mag = math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        if mag < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def inverse(self) -> 'Quaternion':
        mag_sq = self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2
        if mag_sq < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / mag_sq, -self.x / mag_sq, -self.y / mag_sq, -self.z / mag_sq)


class ComplexMath:
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def add(self, other: 'ComplexMath') -> 'ComplexMath':
        return ComplexMath(self.real + other.real, self.imag + other.imag)

    def sub(self, other: 'ComplexMath') -> 'ComplexMath':
        return ComplexMath(self.real - other.real, self.imag - other.imag)

    def mul(self, other: 'ComplexMath') -> 'ComplexMath':
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return ComplexMath(r, i)

    def div(self, other: 'ComplexMath') -> 'ComplexMath':
        denom = other.real ** 2 + other.imag ** 2
        if denom < 1e-12:
            return ComplexMath(0.0, 0.0)
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return ComplexMath(r, i)

    def conjugate(self) -> 'ComplexMath':
        return ComplexMath(self.real, -self.imag)

    def magnitude(self) -> float:
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self) -> float:
        return math.atan2(self.imag, self.real)


class Tensor2D:
    def __init__(self):
        self.t = [[0.0, 0.0], [0.0, 0.0]]

    def trace(self) -> float:
        return self.t[0][0] + self.t[1][1]


class Tensor3D:
    def __init__(self):
        self.t = [[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]

    def trace(self) -> float:
        return self.t[0][0] + self.t[1][1] + self.t[2][2]

    def add(self, other: 'Tensor3D') -> 'Tensor3D':
        res = Tensor3D()
        for i in range(3):
            for j in range(3):
                res.t[i][j] = self.t[i][j] + other.t[i][j]
        return res

    def multiply_vector(self, vec: Vector3D) -> Vector3D:
        x = self.t[0][0] * vec.x + self.t[0][1] * vec.y + self.t[0][2] * vec.z
        y = self.t[1][0] * vec.x + self.t[1][1] * vec.y + self.t[1][2] * vec.z
        z = self.t[2][0] * vec.x + self.t[2][1] * vec.y + self.t[2][2] * vec.z
        return Vector3D(x, y, z)


class JonesVector:
    def __init__(self, ex: ComplexMath, ey: ComplexMath):
        self.ex = ex
        self.ey = ey

    def intensity(self) -> float:
        return self.ex.magnitude() ** 2 + self.ey.magnitude() ** 2

    def normalize(self) -> 'JonesVector':
        intensity = self.intensity()
        if intensity < 1e-12:
            return JonesVector(ComplexMath(0.0, 0.0), ComplexMath(0.0, 0.0))
        factor = math.sqrt(intensity)
        return JonesVector(
            ComplexMath(self.ex.real / factor, self.ex.imag / factor),
            ComplexMath(self.ey.real / factor, self.ey.imag / factor)
        )


class JonesMatrix:
    def __init__(self, j00: ComplexMath, j01: ComplexMath, j10: ComplexMath, j11: ComplexMath):
        self.j00 = j00
        self.j01 = j01
        self.j10 = j10
        self.j11 = j11

    def multiply_vector(self, vec: JonesVector) -> JonesVector:
        nx = self.j00.mul(vec.ex).add(self.j01.mul(vec.ey))
        ny = self.j10.mul(vec.ex).add(self.j11.mul(vec.ey))
        return JonesVector(nx, ny)

    def multiply_matrix(self, other: 'JonesMatrix') -> 'JonesMatrix':
        n00 = self.j00.mul(other.j00).add(self.j01.mul(other.j10))
        n01 = self.j00.mul(other.j01).add(self.j01.mul(other.j11))
        n10 = self.j10.mul(other.j00).add(self.j11.mul(other.j10))
        n11 = self.j10.mul(other.j01).add(self.j11.mul(other.j11))
        return JonesMatrix(n00, n01, n10, n11)


class StokesVector:
    def __init__(self, s0: float, s1: float, s2: float, s3: float):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def degree_of_polarization(self) -> float:
        if self.s0 < 1e-12:
            return 0.0
        return math.sqrt(self.s1 ** 2 + self.s2 ** 2 + self.s3 ** 2) / self.s0


class MuellerMatrix:
    def __init__(self):
        self.m = [[0.0 for _ in range(4)] for _ in range(4)]

    def set_identity(self):
        for i in range(4):
            for j in range(4):
                self.m[i][j] = 1.0 if i == j else 0.0

    def multiply_vector(self, vec: StokesVector) -> StokesVector:
        v_arr = [vec.s0, vec.s1, vec.s2, vec.s3]
        res = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            for j in range(4):
                res[i] += self.m[i][j] * v_arr[j]
        return StokesVector(res[0], res[1], res[2], res[3])


class RungeKutta4:
    def __init__(self, dt: float):
        self.dt = dt

    def step(self, state: np.ndarray, derivative_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        k1 = derivative_func(state)
        k2 = derivative_func(state + 0.5 * self.dt * k1)
        k3 = derivative_func(state + 0.5 * self.dt * k2)
        k4 = derivative_func(state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ThermodynamicsEngine:
    def __init__(self, t: float, p: float, v: float, n: float):
        self.t = t
        self.p = p
        self.v = v
        self.n = n
        self.r = 8.314

    def ideal_gas_pressure(self) -> float:
        if self.v < 1e-12:
            return 0.0
        return (self.n * self.r * self.t) / self.v

    def ideal_gas_volume(self) -> float:
        if self.p < 1e-12:
            return 0.0
        return (self.n * self.r * self.t) / self.p

    def ideal_gas_temperature(self) -> float:
        if self.n < 1e-12:
            return 0.0
        return (self.p * self.v) / (self.n * self.r)

    def isothermal_work(self, v_final: float) -> float:
        if self.v < 1e-12 or v_final < 1e-12:
            return 0.0
        return self.n * self.r * self.t * math.log(v_final / self.v)

    def adiabatic_work(self, v_final: float, gamma: float) -> float:
        if self.v < 1e-12 or v_final < 1e-12 or abs(gamma - 1.0) < 1e-12:
            return 0.0
        p_final = self.p * (self.v / v_final) ** gamma
        return (self.p * self.v - p_final * v_final) / (gamma - 1.0)


class FluidDynamicsSolver:
    def __init__(self, density: float, viscosity: float):
        self.density = density
        self.viscosity = viscosity

    def reynolds_number(self, velocity: float, length: float) -> float:
        if self.viscosity < 1e-12:
            return 0.0
        return (self.density * velocity * length) / self.viscosity

    def drag_force(self, drag_coefficient: float, area: float, velocity: float) -> float:
        return 0.5 * self.density * velocity ** 2 * drag_coefficient * area

    def terminal_velocity(self, mass: float, drag_coefficient: float, area: float, g: float = 9.81) -> float:
        if self.density < 1e-12 or drag_coefficient < 1e-12 or area < 1e-12:
            return 0.0
        return math.sqrt((2.0 * mass * g) / (self.density * area * drag_coefficient))

    def dynamic_pressure(self, velocity: float) -> float:
        return 0.5 * self.density * velocity ** 2

    def bernoulli_constant(self, static_pressure: float, velocity: float, height: float, g: float = 9.81) -> float:
        return static_pressure + self.dynamic_pressure(velocity) + self.density * g * height


class LorentzForceEngine:
    def __init__(self, q: float, m: float):
        self.q = q
        self.m = m

    def compute_acceleration(self, velocity: Vector3D, e_field: Vector3D, b_field: Vector3D) -> Vector3D:
        magnetic_force = velocity.cross(b_field)
        total_force = e_field.add(magnetic_force).mul(self.q)
        return total_force.div(self.m)


class ElectromagneticField:
    def __init__(self, e_vec: Vector3D, b_vec: Vector3D):
        self.electric = e_vec
        self.magnetic = b_vec

    def poynting_vector(self) -> Vector3D:
        return self.electric.cross(self.magnetic)

    def energy_density(self, epsilon: float, mu: float) -> float:
        return 0.5 * (epsilon * self.electric.mag_sq() + (1.0 / mu) * self.magnetic.mag_sq())


class DipoleRadiationEngine:
    def __init__(self, dipole_moment: float, frequency: float, c: float = 3e8):
        self.dipole_moment = dipole_moment
        self.frequency = frequency
        self.c = c
        self.omega = 2 * math.pi * frequency
        self.k = self.omega / c

    def intensity_at_angle(self, theta: float, distance: float) -> float:
        if distance < 1e-12:
            return 0.0
        amplitude_factor = (self.dipole_moment * self.omega ** 4) / (32 * math.pi ** 2 * self.c ** 3 * 8.854e-12)
        return amplitude_factor * (math.sin(theta) ** 2) / (distance ** 2)


class RelativityEngine:
    def __init__(self, c: float = 299792458.0):
        self.c = c

    def gamma_factor(self, v: float) -> float:
        ratio = v / self.c
        if ratio >= 1.0 or ratio <= -1.0:
            return float('inf')
        return 1.0 / math.sqrt(1.0 - ratio ** 2)

    def time_dilation(self, proper_time: float, v: float) -> float:
        return proper_time * self.gamma_factor(v)

    def length_contraction(self, proper_length: float, v: float) -> float:
        return proper_length / self.gamma_factor(v)

    def relativistic_momentum(self, m0: float, v: float) -> float:
        return self.gamma_factor(v) * m0 * v

    def total_energy(self, m0: float, v: float) -> float:
        return self.gamma_factor(v) * m0 * self.c ** 2


class CelestialMechanics:
    def __init__(self, G: float = 6.67430e-11):
        self.G = G

    def gravitational_force(self, m1: float, m2: float, r: float) -> float:
        if r < 1e-12:
            return 0.0
        return self.G * (m1 * m2) / (r ** 2)

    def orbital_velocity(self, M: float, r: float) -> float:
        if r < 1e-12:
            return 0.0
        return math.sqrt(self.G * M / r)

    def escape_velocity(self, M: float, r: float) -> float:
        if r < 1e-12:
            return 0.0
        return math.sqrt(2.0 * self.G * M / r)

    def orbital_period(self, M: float, semi_major_axis: float) -> float:
        if M < 1e-12:
            return float('inf')
        return 2.0 * math.pi * math.sqrt((semi_major_axis ** 3) / (self.G * M))


class QuantumScattering:
    def __init__(self, hbar: float = 1.054571817e-34, m: float = 9.10938356e-31):
        self.hbar = hbar
        self.m = m

    def wave_number(self, E: float, V: float) -> ComplexMath:
        if E >= V:
            k = math.sqrt(2.0 * self.m * (E - V)) / self.hbar
            return ComplexMath(k, 0.0)
        else:
            kappa = math.sqrt(2.0 * self.m * (V - E)) / self.hbar
            return ComplexMath(0.0, kappa)

    def transmission_coefficient_step(self, E: float, V0: float) -> float:
        if E <= V0:
            return 0.0
        k1 = self.wave_number(E, 0.0).real
        k2 = self.wave_number(E, V0).real
        if k1 + k2 < 1e-12:
            return 0.0
        T = (4.0 * k1 * k2) / ((k1 + k2) ** 2)
        return T

    def reflection_coefficient_step(self, E: float, V0: float) -> float:
        if E <= V0:
            return 1.0
        k1 = self.wave_number(E, 0.0).real
        k2 = self.wave_number(E, V0).real
        if k1 + k2 < 1e-12:
            return 1.0
        R = ((k1 - k2) / (k1 + k2)) ** 2
        return R


class SolidStatePhysics:
    def __init__(self, lattice_constant: float):
        self.a = lattice_constant

    def reciprocal_lattice_vector(self) -> float:
        if self.a < 1e-12:
            return 0.0
        return 2.0 * math.pi / self.a

    def bragg_diffraction_angle(self, n: int, wavelength: float, d_spacing: float) -> float:
        val = (n * wavelength) / (2.0 * d_spacing)
        if val > 1.0 or val < -1.0:
            return 0.0
        return math.asin(val)

    def fermi_energy(self, electron_density: float) -> float:
        hbar = 1.054571817e-34
        m_e = 9.10938356e-31
        return ((hbar ** 2) / (2.0 * m_e)) * (3.0 * math.pi ** 2 * electron_density) ** (2.0 / 3.0)


class StandardModelEngine:
    def __init__(self, higgs_vev: float = 246.22):
        self.v = higgs_vev

    def w_boson_mass(self, g: float) -> float:
        return 0.5 * g * self.v

    def z_boson_mass(self, g: float, g_prime: float) -> float:
        return 0.5 * self.v * math.sqrt(g ** 2 + g_prime ** 2)

    def weinberg_angle(self, g: float, g_prime: float) -> float:
        if g < 1e-12:
            return 0.0
        return math.atan(g_prime / g)


class ColorMath:
    @staticmethod
    def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
        hex_str = str(hex_str).lstrip('#')
        if len(hex_str) == 3:
            hex_str = hex_str[0] * 2 + hex_str[1] * 2 + hex_str[2] * 2
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return r, g, b

    @staticmethod
    def rgb_to_hex(r: float, g: float, b: float) -> str:
        r_int = int(max(0.0, min(1.0, r)) * 255)
        g_int = int(max(0.0, min(1.0, g)) * 255)
        b_int = int(max(0.0, min(1.0, b)) * 255)
        return f"#{r_int:02x}{g_int:02x}{b_int:02x}"

    @staticmethod
    def lerp_color(c1: str, c2: str, t: float) -> str:
        r1, g1, b1 = ColorMath.hex_to_rgb(c1)
        r2, g2, b2 = ColorMath.hex_to_rgb(c2)
        t = max(0.0, min(1.0, t))
        r = r1 + (r2 - r1) * t
        g = g1 + (g2 - g1) * t
        b = b1 + (b2 - b1) * t
        return ColorMath.rgb_to_hex(r, g, b)


class OpticsPhysics:
    @staticmethod
    def calculate_fresnel_p(theta_i: float, n1: float, n2: float) -> Tuple[float, float]:
        theta_t_arg = (n1 / n2) * math.sin(theta_i)
        if abs(theta_t_arg) >= 1.0:
            term1 = complex(n2 * math.cos(theta_i), 0.0)
            theta_t_comp = cmath.asin(theta_t_arg)
            term2 = complex(n1, 0.0) * cmath.cos(theta_t_comp)
            r_p = (term1 - term2) / (term1 + term2)
            R_p = abs(r_p) ** 2
            t_p = complex(2.0, 0.0) * term1 / (term1 + term2)
            factor = complex(n2, 0.0) * cmath.cos(theta_t_comp) / (complex(n1, 0.0) * complex(math.cos(theta_i), 0.0))
            T_p = abs(t_p) ** 2 * factor.real
            return R_p, T_p
        else:
            theta_t = math.asin(theta_t_arg)
            term1 = n2 * math.cos(theta_i)
            term2 = n1 * math.cos(theta_t)
            R_p = ((term1 - term2) / (term1 + term2)) ** 2
            t_p = 2.0 * term1 / (term1 + term2)
            T_p = (t_p ** 2) * term2 / term1
            return R_p, T_p


class RealisticEnergyBeam(VGroup):
    def __init__(self, intensity: float = 1.0, color: str = "#00FFFF", **kwargs):
        super().__init__(**kwargs)
        self.beam_intensity = intensity
        self.base_widths = [30.0, 18.0, 8.0, 3.0, 1.2]
        self.layer_ops = [0.03, 0.08, 0.25, 0.6, 0.95]
        self.lines = VGroup()
        for w, op in zip(self.base_widths, self.layer_ops):
            self.lines.add(Line(ORIGIN, RIGHT, color=color, stroke_width=w))
        self.lines[-1].set_color(WHITE)
        self.add(self.lines)

        self.particle_num = 18
        self.particles = VGroup()
        self.particle_ts = []
        for i in range(self.particle_num):
            p = Line(ORIGIN, RIGHT, color=WHITE, stroke_width=2.5)
            self.particles.add(p)
            self.particle_ts.append(float(i) / self.particle_num)
        self.add(self.particles)

        self.start_pt = ORIGIN
        self.end_pt = RIGHT
        self.total_time = 0.0
        self.alpha_mult = 1.0

        self.set_beam_intensity(self.beam_intensity)

    def set_beam_intensity(self, intensity: float):
        self.beam_intensity = max(0.0, min(1.0, intensity))

        self.alpha_mult = max(0.0, min(1.0, intensity))
        if self.alpha_mult < 1e-3:
            for line in self.lines:
                line.set_stroke(opacity=0.0)
            for seg in self.particles:
                seg.set_stroke(opacity=0.0)
        else:
            self.lines[-1].set_stroke(opacity=self.layer_ops[-1] * self.alpha_mult)

        width_boost = max(1.0, 1.5 * intensity)
        for i, line in enumerate(self.lines[:-1]):
            line.set_stroke(width=self.base_widths[i] * width_boost)

    def put_start_and_end_on(self, start: np.ndarray, end: np.ndarray):
        self.start_pt = start
        self.end_pt = end

        vec = end - start
        self.length = np.linalg.norm(vec)
        if self.length < 1e-12:
            self.dir_vec = np.array([1.0, 0.0, 0.0])
        else:
            self.dir_vec = vec / self.length

        for line in self.lines:
            line.put_start_and_end_on(start, end)

    def advance_flow(self, dt: float, global_speed: float = 1.8):
        if self.length < 1e-3 or self.alpha_mult < 1e-3:
            for p in self.particles:
                p.set_stroke(opacity=0.0)
                p.put_start_and_end_on(ORIGIN, ORIGIN + RIGHT * 0.0001)
            return

        self.total_time += dt
        pulse = 0.85 + 0.15 * math.sin(self.total_time * 20.0)

        for i, (line, op) in enumerate(zip(self.lines[:-1], self.layer_ops[:-1])):
            line.set_stroke(opacity=op * pulse * self.alpha_mult)

        for i, seg in enumerate(self.particles):
            t = self.particle_ts[i]
            t += dt * global_speed * random.uniform(0.8, 1.2)
            if t > 1.0:
                t -= 1.0
            self.particle_ts[i] = t

            seg_len = random.uniform(0.15, 0.4)
            start_dist = t * self.length
            end_dist = start_dist + seg_len

            if end_dist > self.length:
                end_dist = self.length
                start_dist = max(0.0, end_dist - seg_len)

            if start_dist >= end_dist - 1e-4:
                seg.set_stroke(opacity=0.0)
                seg.put_start_and_end_on(self.start_pt, self.start_pt + self.dir_vec * 0.0001)
            else:
                p1 = self.start_pt + self.dir_vec * start_dist
                p2 = self.start_pt + self.dir_vec * end_dist
                seg.put_start_and_end_on(p1, p2)

                fade_in = min(1.0, start_dist / 0.3)
                fade_out = min(1.0, (self.length - end_dist) / 0.3)
                final_op = random.uniform(0.6, 0.95) * fade_in * fade_out * pulse * self.alpha_mult
                seg.set_stroke(opacity=final_op)


class EnergyArrow(VGroup):
    def __init__(self, color: str, **kwargs):
        super().__init__(**kwargs)
        self.base_color = color
        p1g, p2g, p3g = np.array([-0.3, -0.2, 0]), np.array([-0.3, 0.2, 0]), np.array([0.3, 0, 0])
        p1c, p2c, p3c = np.array([-0.25, -0.15, 0]), np.array([-0.25, 0.15, 0]), np.array([0.25, 0, 0])

        self.poly_glow = Polygon(p1g, p2g, p3g, fill_color=color, fill_opacity=0.4, stroke_width=0)
        self.poly_core = Polygon(p1c, p2c, p3c, fill_color=WHITE, fill_opacity=0.9, stroke_width=0)
        self.add(self.poly_glow, self.poly_core)

    def update_pose(self, start_pt: np.ndarray, end_pt: np.ndarray, alpha: float, visible_boost: float,
                    width_factor: float):
        if visible_boost < 0.001 or width_factor < 1e-3:
            self.poly_glow.set_fill(opacity=0)
            self.poly_core.set_fill(opacity=0)
            return

        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 0.001:
            self.poly_glow.set_fill(opacity=0)
            self.poly_core.set_fill(opacity=0)
            return

        unit_dir = direction / length
        angle = math.atan2(unit_dir[1], unit_dir[0])
        pos = start_pt + direction * alpha

        c = math.cos(angle)
        s = math.sin(angle)

        def rot(p):
            return np.array([p[0] * c - p[1] * s, p[0] * s + p[1] * c, 0])

        p1g, p2g, p3g = np.array([-0.3, -0.2, 0]), np.array([-0.3, 0.2, 0]), np.array([0.3, 0, 0])
        p1c, p2c, p3c = np.array([-0.25, -0.15, 0]), np.array([-0.25, 0.15, 0]), np.array([0.25, 0, 0])

        self.poly_glow.become(Polygon(rot(p1g) + pos, rot(p2g) + pos, rot(p3g) + pos, fill_color=self.base_color,
                                      fill_opacity=0.4 * visible_boost, stroke_width=0))
        self.poly_core.become(
            Polygon(rot(p1c) + pos, rot(p2c) + pos, rot(p3c) + pos, fill_color=WHITE, fill_opacity=0.9 * visible_boost,
                    stroke_width=0))


class RealisticLightSource(VGroup):
    def __init__(self, color: str, **kwargs):
        super().__init__(**kwargs)
        self.core = Dot(radius=0.12, color=WHITE)
        self.halo1 = Dot(radius=0.25, color=color, fill_opacity=0.6)
        self.halo2 = Dot(radius=0.50, color=color, fill_opacity=0.20)
        self.halo3 = Dot(radius=0.85, color=color, fill_opacity=0.06)
        self.add(self.halo3, self.halo2, self.halo1, self.core)


class MacroBrewsterTransmissionScene(Scene):
    def construct(self):
        tracker = ValueTracker(20 * DEGREES)

        col_water = "#002244"
        col_air = "#112233"
        n_air = 1.00
        n_glass = 1.50

        theta_B = math.atan(n_glass / n_air)
        ray_len = 5.0

        grid = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": "#2A3A4A",
                "stroke_width": 1.5,
                "stroke_opacity": 0.25
            }
        )
        self.add(grid)

        air_rect = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0)
        air_rect.set_fill(color=col_air, opacity=0.8)
        air_rect.next_to(ORIGIN, UP, buff=0)

        glass_rect = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0)
        glass_rect.set_fill(color=col_water, opacity=0.8)
        glass_rect.next_to(ORIGIN, DOWN, buff=0)

        boundary = Line(LEFT * 8, RIGHT * 8, color="#88AACC", stroke_width=3)
        normal = DashedLine(DOWN * 5, UP * 5, color=WHITE, dash_length=0.12, stroke_opacity=0.5)

        air_text = Text("空气 (n=1.00)", font_size=28, weight=BOLD, color=WHITE, font="Microsoft YaHei")
        air_text.to_corner(UL).shift(RIGHT * 0.5 + DOWN * 0.5)

        glass_text = Text("玻璃 (n=1.50)", font_size=28, weight=BOLD, color=WHITE, font="Microsoft YaHei")
        glass_text.to_corner(DL).shift(RIGHT * 0.5 + UP * 0.5)

        col_beams = "#66FFFF"

        inc_ray = RealisticEnergyBeam(intensity=1.0, color=col_beams)
        ref_ray = RealisticEnergyBeam(intensity=1.0, color=col_beams)
        tra_ray = RealisticEnergyBeam(intensity=1.0, color=col_beams)

        arr_inc = EnergyArrow(color=col_beams)
        arr_ref = EnergyArrow(color=col_beams)
        arr_tra = EnergyArrow(color=col_beams)

        src = RealisticLightSource(color=col_beams)

        info_panel_bg = RoundedRectangle(corner_radius=0.15, width=4.5, height=2.2, color="#445566",
                                         fill_color="#000000", fill_opacity=0.8)
        info_panel_bg.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)

        val_inc = Text("入射角: 00.0°", font_size=20, color=WHITE, font="Microsoft YaHei")
        val_rp = Text("反射率 (Rp): 0.0%", font_size=20, color=col_beams, font="Microsoft YaHei")
        val_tb = Text(f"布儒斯特角: {theta_B * 180 / math.pi:.1f}°", font_size=20, color=YELLOW, font="Microsoft YaHei")

        text_group = VGroup(val_inc, val_rp, val_tb).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        text_group.move_to(info_panel_bg.get_center())
        info_panel = VGroup(info_panel_bg, text_group)

        val_inc_ref = val_inc.get_left()
        val_rp_ref = val_rp.get_left()

        arc_mob = VMobject()
        theta_label = MathTex(r"\theta_1", font_size=36, color=WHITE)

        def update_inc(mob):
            th = tracker.get_value()
            pt = np.array([-ray_len * math.sin(th), ray_len * math.cos(th), 0])
            mob.put_start_and_end_on(pt, ORIGIN)

        def update_inc_arr(mob):
            th = tracker.get_value()
            pt = np.array([-ray_len * math.sin(th), ray_len * math.cos(th), 0])
            mob.update_pose(pt, ORIGIN, 0.45, 1.0, 1.0)
            src.move_to(pt)

        def update_ref(mob):
            th = tracker.get_value()
            R_p, T_p = OpticsPhysics.calculate_fresnel_p(th, n_air, n_glass)
            pt = np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
            mob.put_start_and_end_on(ORIGIN, pt)
            boost = min(1.0, R_p * 15.0)
            mob.set_beam_intensity(boost)

        def update_ref_arr(mob):
            th = tracker.get_value()
            R_p, T_p = OpticsPhysics.calculate_fresnel_p(th, n_air, n_glass)
            pt = np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
            boost = min(1.0, R_p * 15.0)
            mob.update_pose(ORIGIN, pt, 0.55, boost, boost)

        def update_tra(mob):
            th = tracker.get_value()
            R_p, T_p = OpticsPhysics.calculate_fresnel_p(th, n_air, n_glass)
            sin_t = (n_air / n_glass) * math.sin(th)
            th_t = math.asin(sin_t)
            pt = np.array([ray_len * math.sin(th_t), -ray_len * math.cos(th_t), 0])
            mob.put_start_and_end_on(ORIGIN, pt)
            mob.set_beam_intensity(T_p)

        def update_tra_arr(mob):
            th = tracker.get_value()
            R_p, T_p = OpticsPhysics.calculate_fresnel_p(th, n_air, n_glass)
            sin_t = (n_air / n_glass) * math.sin(th)
            th_t = math.asin(sin_t)
            pt = np.array([ray_len * math.sin(th_t), -ray_len * math.cos(th_t), 0])
            mob.update_pose(ORIGIN, pt, 0.55, 1.0, T_p)

        def update_arc(mob):
            th = tracker.get_value()
            new_arc = Arc(radius=1.8, start_angle=math.pi / 2, angle=th, color=WHITE, stroke_width=3.5)
            new_arc.add_tip(tip_length=0.25, tip_width=0.25)
            mob.become(new_arc)

        def update_theta_label(mob):
            th = tracker.get_value()
            r = 2.4
            bisector = math.pi / 2 + th / 2
            pos = np.array([r * math.cos(bisector), r * math.sin(bisector), 0])
            mob.become(MathTex(f"\\theta_1 = {th * 180 / math.pi:.1f}^\\circ", font_size=32, color=WHITE).move_to(pos))

        def update_val_inc(mob):
            th = tracker.get_value()
            deg = th * 180 / math.pi
            new_text = Text(f"入射角: {deg:.1f}°", font_size=20, color=WHITE, font="Microsoft YaHei")
            new_text.move_to(val_inc_ref, aligned_edge=LEFT)
            mob.become(new_text)

        def update_val_rp(mob):
            th = tracker.get_value()
            R_p, T_p = OpticsPhysics.calculate_fresnel_p(th, n_air, n_glass)
            rp_percent = R_p * 100.0

            if rp_percent < 0.05:
                rp_str = "0.00"
                col = RED
            else:
                rp_str = f"{rp_percent:.2f}"
                col = col_beams

            new_text = Text(f"反射率 (Rp): {rp_str}%", font_size=20, color=col, font="Microsoft YaHei")
            new_text.move_to(val_rp_ref, aligned_edge=LEFT)
            mob.become(new_text)

        def inc_flow_updater(mob, dt):
            mob.advance_flow(dt, global_speed=2.2)

        def ref_flow_updater(mob, dt):
            mob.advance_flow(dt, global_speed=2.2)

        def tra_flow_updater(mob, dt):
            mob.advance_flow(dt, global_speed=2.2)

        update_inc(inc_ray)
        update_inc_arr(arr_inc)
        update_ref(ref_ray)
        update_ref_arr(arr_ref)
        update_tra(tra_ray)
        update_tra_arr(arr_tra)
        update_arc(arc_mob)
        update_theta_label(theta_label)
        update_val_inc(val_inc)
        update_val_rp(val_rp)

        self.play(
            FadeIn(air_rect), FadeIn(glass_rect),
            Create(boundary), Create(normal),
            Write(air_text), Write(glass_text),
            FadeIn(info_panel),
            run_time=1.5
        )

        self.play(
            FadeIn(src),
            FadeIn(inc_ray), FadeIn(arr_inc),
            FadeIn(ref_ray), FadeIn(arr_ref),
            FadeIn(tra_ray), FadeIn(arr_tra),
            Create(arc_mob), FadeIn(theta_label),
            run_time=1.5
        )

        inc_ray.add_updater(update_inc)
        arr_inc.add_updater(update_inc_arr)
        ref_ray.add_updater(update_ref)
        arr_ref.add_updater(update_ref_arr)
        tra_ray.add_updater(update_tra)
        arr_tra.add_updater(update_tra_arr)
        arc_mob.add_updater(update_arc)
        theta_label.add_updater(update_theta_label)
        val_inc.add_updater(update_val_inc)
        val_rp.add_updater(update_val_rp)

        inc_ray.add_updater(inc_flow_updater)
        ref_ray.add_updater(ref_flow_updater)
        tra_ray.add_updater(tra_flow_updater)

        self.play(
            tracker.animate.set_value(theta_B),
            run_time=4.0,
            rate_func=rate_functions.ease_in_out_sine
        )

        self.wait(2.0)

        self.play(
            tracker.animate.set_value(80 * DEGREES),
            run_time=3.5,
            rate_func=rate_functions.ease_in_out_sine
        )

        self.wait(1)

        summary_bg = RoundedRectangle(corner_radius=0.2, width=9.5, height=1.2, color=YELLOW, fill_color=BLACK,
                                      fill_opacity=0.85)
        summary_bg.to_edge(DOWN, buff=0.5)
        summary_text = Text("p偏振光在布儒斯特角入射时，反射率为零，实现全透射", font_size=26, color=YELLOW,
                            font="Microsoft YaHei")
        summary_text.move_to(summary_bg.get_center())

        final_summary = VGroup(summary_bg, summary_text)

        self.play(FadeIn(final_summary, shift=UP * 0.5), run_time=1.2)
        self.wait(4)

        inc_ray.remove_updater(inc_flow_updater)
        ref_ray.remove_updater(ref_flow_updater)
        tra_ray.remove_updater(tra_flow_updater)


if __name__ == "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = MacroBrewsterTransmissionScene()
        scene.render()