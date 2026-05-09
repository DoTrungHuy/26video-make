from manim import *
import numpy as np
import math
import random
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
    def calculate_fresnel_p(theta_i: float, n1: float, n2: float) -> float:
        sin_t_sq = (n1 / n2) ** 2 * (math.sin(theta_i) ** 2)
        if sin_t_sq >= 1.0:
            return 1.0
        theta_t = math.asin(math.sqrt(sin_t_sq))
        num = n2 * math.cos(theta_i) - n1 * math.cos(theta_t)
        den = n2 * math.cos(theta_i) + n1 * math.cos(theta_t)
        return (num / den) ** 2

    @staticmethod
    def calculate_evanescent_decay(theta_i: float, n1: float, n2: float, wavelength: float) -> float:
        sin_i = math.sin(theta_i)
        if (n1 / n2) * sin_i <= 1.0:
            return 0.0
        alpha = (2 * math.pi / wavelength) * n2 * math.sqrt((n1 / n2) ** 2 * sin_i ** 2 - 1.0)
        return alpha


class GlowLine(VGroup):
    def __init__(self, start, end, base_color, **kwargs):
        super().__init__(**kwargs)
        self.layer_data = [
            {"width": 38.0, "opacity": 0.015},
            {"width": 24.0, "opacity": 0.04},
            {"width": 12.0, "opacity": 0.12},
            {"width": 5.0, "opacity": 0.40},
            {"width": 1.5, "opacity": 1.00}
        ]
        self.lines = VGroup()
        for ld in self.layer_data:
            self.lines.add(
                Line(
                    start,
                    end,
                    color=base_color,
                    stroke_width=ld["width"],
                    stroke_opacity=ld["opacity"]
                )
            )
        self.add(self.lines)

    def put_start_and_end_on(self, start, end):
        for line in self.lines:
            line.put_start_and_end_on(start, end)

    def set_glow_opacity(self, alpha):
        alpha = max(0.0, min(1.0, alpha))
        for line, ld in zip(self.lines, self.layer_data):
            line.set_stroke(opacity=ld["opacity"] * alpha)


class RayArrow(Polygon):
    def __init__(self, color, **kwargs):
        super().__init__(
            np.array([-0.20, -0.15, 0]),
            np.array([-0.20, 0.15, 0]),
            np.array([0.20, 0, 0]),
            color=color,
            fill_opacity=1.0,
            stroke_width=0,
            **kwargs
        )
        self.base_color = color

    def update_pose(self, start_pt, end_pt, alpha, visible_boost):
        if visible_boost < 0.001:
            self.set_fill(opacity=0)
            return
        self.set_fill(opacity=visible_boost)
        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 0.001:
            self.set_fill(opacity=0)
            return
        unit_dir = direction / length
        angle = math.atan2(unit_dir[1], unit_dir[0])
        pos = start_pt + direction * alpha

        p1 = np.array([-0.18, -0.12, 0])
        p2 = np.array([-0.18, 0.12, 0])
        p3 = np.array([0.18, 0.00, 0])

        c = math.cos(angle)
        s = math.sin(angle)

        def rot(p):
            return np.array([p[0] * c - p[1] * s, p[0] * s + p[1] * c, 0])

        self.become(
            Polygon(
                rot(p1) + pos,
                rot(p2) + pos,
                rot(p3) + pos,
                color=self.base_color,
                fill_opacity=visible_boost,
                stroke_width=0
            )
        )


class LightSource(VGroup):
    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
        self.core = Dot(radius=0.10, color=WHITE)
        self.halo1 = Dot(radius=0.20, color=color, fill_opacity=0.7)
        self.halo2 = Dot(radius=0.45, color=color, fill_opacity=0.25)
        self.halo3 = Dot(radius=0.75, color=color, fill_opacity=0.08)
        self.add(self.halo3, self.halo2, self.halo1, self.core)


class SplitScreenTIR(Scene):
    def construct(self):
        tracker = ValueTracker(25 * DEGREES)

        col_water = "#002244"
        col_air = "#112233"
        n_water = 1.33
        n_air = 1.00

        w_half = config.frame_width / 2
        h_half = config.frame_height / 2
        ray_len = 3.2

        def build_system(offset_x, n_top, n_bot, color_top, color_bot, label_top, label_bot):
            center = np.array([offset_x, 0, 0])

            rect_top = Rectangle(width=w_half, height=h_half, stroke_width=0, fill_color=color_top, fill_opacity=0.6)
            rect_top.move_to(center + UP * h_half / 2)

            rect_bot = Rectangle(width=w_half, height=h_half, stroke_width=0, fill_color=color_bot, fill_opacity=0.6)
            rect_bot.move_to(center + DOWN * h_half / 2)

            boundary = Line(center + LEFT * w_half / 2, center + RIGHT * w_half / 2, color="#88AACC", stroke_width=3)
            normal = DashedLine(center + DOWN * 3.5, center + UP * 3.5, color=WHITE, dash_length=0.12,
                                stroke_opacity=0.5)

            t_top = Text(label_top, font_size=20, color=WHITE, font="Microsoft YaHei").move_to(
                center + UP * (h_half - 0.4))
            t_bot = Text(label_bot, font_size=20, color=WHITE, font="Microsoft YaHei").move_to(
                center + DOWN * (h_half - 0.4))

            inc_ray = GlowLine(ORIGIN, ORIGIN, base_color="#FF2200")
            ref_ray = GlowLine(ORIGIN, ORIGIN, base_color="#FF8800")
            tra_ray = GlowLine(ORIGIN, ORIGIN, base_color="#00AAFF")

            arr_inc = RayArrow(color="#FF2200")
            arr_ref = RayArrow(color="#FF8800")
            arr_tra = RayArrow(color="#00AAFF")

            src = LightSource(color="#FF2200")

            arc_mob = VMobject()
            theta_label = MathTex(r"\theta_1", font_size=28, color=WHITE)

            def update_inc(mob):
                th = tracker.get_value()
                pt = center + np.array([-ray_len * math.sin(th), ray_len * math.cos(th), 0])
                mob.put_start_and_end_on(pt, center)

            def update_inc_arr(mob):
                th = tracker.get_value()
                pt = center + np.array([-ray_len * math.sin(th), ray_len * math.cos(th), 0])
                mob.update_pose(pt, center, 0.45, 1.0)
                src.move_to(pt)

            def update_ref(mob):
                th = tracker.get_value()
                val = (n_top / n_bot) * math.sin(th)
                pt = center + np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
                mob.put_start_and_end_on(center, pt)
                boost = min(1.0, 0.15 + 0.85 * (val ** 6)) if val < 1.0 else 1.0
                mob.set_glow_opacity(boost)

            def update_ref_arr(mob):
                th = tracker.get_value()
                val = (n_top / n_bot) * math.sin(th)
                pt = center + np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
                boost = min(1.0, 0.15 + 0.85 * (val ** 6)) if val < 1.0 else 1.0
                mob.update_pose(center, pt, 0.55, boost)

            def update_tra(mob):
                th = tracker.get_value()
                val = (n_top / n_bot) * math.sin(th)
                if val >= 0.9999:
                    safe_pt = center + np.array([0.001, 0.001, 0])
                    mob.put_start_and_end_on(center, safe_pt)
                    mob.set_glow_opacity(0.0)
                else:
                    th_t = math.asin(val)
                    pt = center + np.array([ray_len * math.sin(th_t), -ray_len * math.cos(th_t), 0])
                    mob.put_start_and_end_on(center, pt)
                    fade = max(0.0, 1.0 - (val ** 18))
                    mob.set_glow_opacity(fade)

            def update_tra_arr(mob):
                th = tracker.get_value()
                val = (n_top / n_bot) * math.sin(th)
                if val >= 0.9999:
                    safe_pt = center + np.array([0.001, 0.001, 0])
                    mob.update_pose(center, safe_pt, 0.5, 0.0)
                else:
                    th_t = math.asin(val)
                    pt = center + np.array([ray_len * math.sin(th_t), -ray_len * math.cos(th_t), 0])
                    fade = max(0.0, 1.0 - (val ** 18))
                    mob.update_pose(center, pt, 0.55, fade)

            def update_arc(mob):
                th = tracker.get_value()
                new_arc = Arc(arc_center=center, radius=1.2, start_angle=math.pi / 2, angle=th, color=WHITE,
                              stroke_width=2.5)
                new_arc.add_tip(tip_length=0.15, tip_width=0.15)
                mob.become(new_arc)

            def update_theta_label(mob):
                th = tracker.get_value()
                r = 1.7
                bisector = math.pi / 2 + th / 2
                pos = center + np.array([r * math.cos(bisector), r * math.sin(bisector), 0])
                mob.become(
                    MathTex(f"\\theta_1 = {th * 180 / math.pi:.1f}^\\circ", font_size=28, color=WHITE).move_to(pos))

            update_inc(inc_ray)
            update_inc_arr(arr_inc)
            update_ref(ref_ray)
            update_ref_arr(arr_ref)
            update_tra(tra_ray)
            update_tra_arr(arr_tra)
            update_arc(arc_mob)
            update_theta_label(theta_label)

            def attach_updaters():
                inc_ray.add_updater(update_inc)
                arr_inc.add_updater(update_inc_arr)
                ref_ray.add_updater(update_ref)
                arr_ref.add_updater(update_ref_arr)
                tra_ray.add_updater(update_tra)
                arr_tra.add_updater(update_tra_arr)
                arc_mob.add_updater(update_arc)
                theta_label.add_updater(update_theta_label)

            static_grp = VGroup(rect_top, rect_bot, boundary, normal, t_top, t_bot)
            dynamic_grp = VGroup(inc_ray, ref_ray, tra_ray, arr_inc, arr_ref, arr_tra, src, arc_mob, theta_label)

            return static_grp, dynamic_grp, attach_updaters

        left_offset = -w_half / 2
        right_offset = w_half / 2

        left_static, left_dynamic, attach_left = build_system(
            left_offset, n_water, n_air, col_water, col_air,
            "水 (光密介质, n=1.33)", "空气 (光疏介质, n=1.00)"
        )

        right_static, right_dynamic, attach_right = build_system(
            right_offset, n_air, n_water, col_air, col_water,
            "空气 (光疏介质, n=1.00)", "水 (光密介质, n=1.33)"
        )

        divider = Line(UP * config.frame_height / 2, DOWN * config.frame_height / 2, color=WHITE, stroke_width=4)

        self.play(
            FadeIn(left_static),
            FadeIn(right_static),
            Create(divider),
            run_time=1.5
        )

        self.play(
            FadeIn(left_dynamic),
            FadeIn(right_dynamic),
            run_time=1.5
        )

        attach_left()
        attach_right()

        target_angle = 90 * DEGREES
        self.play(
            tracker.animate.set_value(target_angle),
            run_time=6.0,
            rate_func=linear
        )

        self.wait(2)

        summary_bg = RoundedRectangle(corner_radius=0.2, width=8.5, height=1.2, color=YELLOW, fill_color=BLACK,
                                      fill_opacity=0.85)
        summary_bg.to_edge(DOWN, buff=0.5)
        summary_text = Text("全反射的必要条件：光必须从光密介质射向光疏介质", font_size=26, color=YELLOW,
                            font="Microsoft YaHei")
        summary_text.move_to(summary_bg.get_center())

        final_summary = VGroup(summary_bg, summary_text)

        self.play(FadeIn(final_summary, shift=UP * 0.5), run_time=1.2)
        self.wait(3)


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True}):
        scene = SplitScreenTIR()
        scene.render()