# from https://github.com/zhm-real/CurvesGenerator
# with some minor modifications made by Ryota Ishidu

from math import pi, cos, sin, tan, sqrt, atan2, asin, acos, radians, hypot
import numpy as np
import rerun as rr
from dataclasses import dataclass

# parameters initiation
STEP_SIZE = 0.2
MAX_LENGTH = 1000.0


# class for PATH element
# @dataclass(frozen=True)
class RSPath:
    lengths: list[
        float
    ]  # lengths of each part of path (+: forward, -: backward) [float]
    ctypes: list[str]  # type of each part of the path [string]
    L: float  # total path length [float]
    x: list[float]  # final x positions [m]
    y: list[float]  # final y positions [m]
    yaw: list[float]  # final yaw angles [rad]
    directions: list[int]  # forward: 1, backward:-1

    def __init__(self, lengths, ctypes, L, x, y, yaw, directions):
        self.lengths = (
            lengths  # lengths of each part of path (+: forward, -: backward) [float]
        )
        self.ctypes = ctypes  # type of each part of the path [string]
        self.L = L  # total path length [float]
        self.x = x  # final x positions [m]
        self.y = y  # final y positions [m]
        self.yaw = yaw  # final yaw angles [rad]
        self.directions = directions  # forward: 1, backward:-1


@dataclass
class Car:
    RF: float  # [m] distance from rear to vehicle front end of vehicle
    RB: float  # [m] distance from rear to vehicle back end of vehicle
    W: float  # [m] width of vehicle
    WD: float  # [m] distance between left-right wheels
    WB: float  # [m] Wheel base
    TR: float  # [m] Tyre radius
    TW: float  # [m] Tyre width


def calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=step_size)

    minL = paths[0].L
    mini = 0

    for i in range(len(paths)):
        if paths[i].L <= minL:
            minL, mini = paths[i].L, i

    return paths[mini]


def calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc)
    for path in paths:
        x, y, yaw, directions = generate_local_course(
            path.L, path.lengths, path.ctypes, maxc, step_size * maxc
        )

        # convert global coordinate
        path.x = [cos(-q0[2]) * ix + sin(-q0[2]) * iy + q0[0] for (ix, iy) in zip(x, y)]
        path.y = [
            -sin(-q0[2]) * ix + cos(-q0[2]) * iy + q0[1] for (ix, iy) in zip(x, y)
        ]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        path.lengths = [l / maxc for l in path.lengths]
        path.L = path.L / maxc

    return paths


def set_path(
    paths: list[RSPath], lengths: list[float], ctypes: list[str]
) -> list[RSPath]:
    # check same path exist
    for path_e in paths:
        if path_e.ctypes == ctypes:
            if sum([x - y for x, y in zip(path_e.lengths, lengths)]) <= 0.01:
                return paths  # not insert path

    total_l = sum([abs(i) for i in lengths])

    if total_l >= MAX_LENGTH:
        return paths

    assert total_l >= 0.01
    path = RSPath(lengths, ctypes, total_l, [], [], [], [])
    paths.append(path)

    return paths


def LSL(x, y, phi):
    u, t = R(x - sin(phi), y - 1.0 + cos(phi))

    if t >= 0.0:
        v = M(phi - t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LSR(x, y, phi):
    u1, t1 = R(x + sin(phi), y - 1.0 - cos(phi))
    u1 = u1**2

    if u1 >= 4.0:
        u = sqrt(u1 - 4.0)
        theta = atan2(2.0, u)
        t = M(t1 + theta)
        v = M(t - phi)

        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRL(x, y, phi):
    u1, t1 = R(x - sin(phi), y - 1.0 + cos(phi))

    if u1 <= 4.0:
        u = -2.0 * asin(0.25 * u1)
        t = M(t1 + 0.5 * u + pi)
        v = M(phi - t + u)

        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def SCS(x, y, phi, paths):
    flag, t, u, v = SLS(x, y, phi)

    if flag:
        paths = set_path(paths, [t, u, v], ["S", "L", "S"])

    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])

    return paths


def SLS(x, y, phi):
    phi = M(phi)

    if y > 0.0 and 0.0 < phi < pi * 0.99:
        xd = -y / tan(phi) + x
        t = xd - tan(phi / 2.0)
        u = phi
        v = sqrt((x - xd) ** 2 + y**2) - tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < pi * 0.99:
        xd = -y / tan(phi) + x
        t = xd - tan(phi / 2.0)
        u = phi
        v = -sqrt((x - xd) ** 2 + y**2) - tan(phi / 2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CSC(x, y, phi, paths):
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "L"])

    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "L"])

    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])

    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])

    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "R"])

    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "R"])

    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "L"])

    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "L"])

    return paths


def CCC(x, y, phi, paths):
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "R", "L"])

    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "R", "L"])

    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "L", "R"])

    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "L", "R"])

    # backwards
    xb = x * cos(phi) + y * sin(phi)
    yb = x * sin(phi) - y * cos(phi)

    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["L", "R", "L"])

    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["L", "R", "L"])

    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "L", "R"])

    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "L", "R"])

    return paths


def calc_tauOmega(u, v, xi, eta, phi):
    delta = M(u - v)
    A = sin(u) - sin(delta)
    B = cos(u) - cos(delta) - 1.0

    t1 = atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (cos(delta) - cos(v) - cos(u)) + 3.0

    if t2 < 0:
        tau = M(t1 + pi)
    else:
        tau = M(t1)

    omega = M(tau - u + v - phi)

    return tau, omega


def LRLRn(x, y, phi):
    xi = x + sin(phi)
    eta = y - 1.0 - cos(phi)
    rho = 0.25 * (2.0 + sqrt(xi * xi + eta * eta))

    if rho <= 1.0:
        u = acos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRLRp(x, y, phi):
    xi = x + sin(phi)
    eta = y - 1.0 - cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0

    if 0.0 <= rho <= 1.0:
        u = -acos(rho)
        if u >= -0.5 * pi:
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCCC(x, y, phi, paths):
    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRn(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["R", "L", "R", "L"])

    return paths


def LRSR(x, y, phi):
    xi = x + sin(phi)
    eta = y - 1.0 - cos(phi)
    rho, theta = R(-eta, xi)

    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = M(t + 0.5 * pi - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRSL(x, y, phi):
    xi = x - sin(phi)
    eta = y - 1.0 + cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        r = sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = M(theta + atan2(r, -2.0))
        v = M(phi - 0.5 * pi - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCSC(x, y, phi, paths):
    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * pi, u, v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * pi, -u, -v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * pi, u, v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * pi, -u, -v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * pi, u, v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * pi, -u, -v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * pi, u, v], ["R", "L", "S", "L"])

    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * pi, -u, -v], ["R", "L", "S", "L"])

    # backwards
    xb = x * cos(phi) + y * sin(phi)
    yb = x * sin(phi) - y * cos(phi)

    flag, t, u, v = LRSL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * pi, t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * pi, -t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * pi, t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * pi, -t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * pi, t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * pi, -t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * pi, t], ["L", "S", "L", "R"])

    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * pi, -t], ["L", "S", "L", "R"])

    return paths


def LRSLR(x, y, phi):
    # formula 8.11 *** TYPO IN PAPER ***
    xi = x + sin(phi)
    eta = y - 1.0 - cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        u = 4.0 - sqrt(rho * rho - 4.0)
        if u <= 0.0:
            t = M(atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = M(t - phi)

            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCSCC(x, y, phi, paths):
    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path(
            paths, [t, -0.5 * pi, u, -0.5 * pi, v], ["L", "R", "S", "L", "R"]
        )

    flag, t, u, v = LRSLR(-x, y, -phi)
    if flag:
        paths = set_path(
            paths, [-t, 0.5 * pi, -u, 0.5 * pi, -v], ["L", "R", "S", "L", "R"]
        )

    flag, t, u, v = LRSLR(x, -y, -phi)
    if flag:
        paths = set_path(
            paths, [t, -0.5 * pi, u, -0.5 * pi, v], ["R", "L", "S", "R", "L"]
        )

    flag, t, u, v = LRSLR(-x, -y, phi)
    if flag:
        paths = set_path(
            paths, [-t, 0.5 * pi, -u, 0.5 * pi, -v], ["R", "L", "S", "R", "L"]
        )

    return paths


def generate_local_course(L, lengths, mode, maxc, step_size):
    point_num = int(L / step_size) + len(lengths) + 3
    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    ll = 0.0
    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]
        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = interpolate(
                ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions
            )
            pd += d
        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = interpolate(
            ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions
        )

    # remove unused data
    while px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    if m == "S":
        px[ind] = ox + l / maxc * cos(oyaw)
        py[ind] = oy + l / maxc * sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = sin(l) / maxc
        if m == "L":
            ldy = (1.0 - cos(l)) / maxc
        elif m == "R":
            ldy = -(1.0 - cos(l)) / maxc

        gdx = cos(-oyaw) * ldx + sin(-oyaw) * ldy
        gdy = -sin(-oyaw) * ldx + cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return px, py, pyaw, directions


def generate_path(q0, q1, maxc):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = cos(q0[2])
    s = sin(q0[2])
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc

    paths = []
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    paths = CCCC(x, y, dth, paths)
    paths = CCSC(x, y, dth, paths)
    paths = CCSCC(x, y, dth, paths)

    return paths


# utils
def pi_2_pi(theta: float) -> float:
    while theta > pi:
        theta -= 2.0 * pi

    while theta < -pi:
        theta += 2.0 * pi

    return theta


def R(x: float, y: float) -> tuple[float, float]:
    """
    Return the polar coordinates (r, theta) of the point (x, y)
    """
    r = hypot(x, y)
    theta = atan2(y, x)

    return r, theta


def M(theta: float) -> float:
    """
    Regulate theta to -pi <= theta < pi
    """
    phi = theta % (2.0 * pi)

    if phi < -pi:
        phi += 2.0 * pi
    if phi > pi:
        phi -= 2.0 * pi

    return phi


def draw_car(x: float, y: float, yaw: float, steer: float, car: Car):
    body = np.array(
        [
            [-car.RB, -car.RB, car.RF, car.RF, -car.RB],
            [car.W / 2, -car.W / 2, -car.W / 2, car.W / 2, car.W / 2],
        ]
    )

    wheel = np.array(
        [
            [-car.TR, -car.TR, car.TR, car.TR, -car.TR],
            [
                car.TW / 4,
                -car.TW / 4,
                -car.TW / 4,
                car.TW / 4,
                car.TW / 4,
            ],
        ]
    )

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])

    Rot2 = np.array([[cos(steer), sin(steer)], [-sin(steer), cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[car.WB], [-car.WD / 2]])
    flWheel += np.array([[car.WB], [car.WD / 2]])
    rrWheel[1, :] -= car.WD / 2
    rlWheel[1, :] += car.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    body = np.dot(Rot1, body)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    body += np.array([[x], [y]])

    rr.log("car/body", rr.LineStrips2D(body.T))
    rr.log("car/fr_wheel", rr.LineStrips2D(frWheel.T))
    rr.log("car/rr_wheel", rr.LineStrips2D(rrWheel.T))
    rr.log("car/fl_wheel", rr.LineStrips2D(flWheel.T))
    rr.log("car/rl_wheel", rr.LineStrips2D(rlWheel.T))
    rr.log("car/dir", rr.Arrows2D(vectors=(np.cos(yaw), np.sin(yaw)), origins=(x, y)))


def main():
    rr.init("reeds-shepp", spawn=True)

    # choose states pairs: (x, y, yaw)
    # simulation-1
    states = [
        (0, 0, 0),
        (10, 10, -90),
        (20, 5, 60),
        (30, 10, 120),
        (35, -5, 30),
        (25, -10, -120),
        (15, -15, 100),
        (0, -10, -90),
    ]

    # simulation-2
    # states = [
    #     (-3, 3, 120),
    #     (10, -7, 30),
    #     (10, 13, 30),
    #     (20, 5, -25),
    #     (35, 10, 180),
    #     (32, -10, 180),
    #     (5, -12, 90),
    # ]
    arrows = np.array(
        list(
            map(
                lambda s: [
                    s[0],
                    s[1],
                    cos(radians(s[2])),
                    sin(radians(s[2])),
                ],
                states,
            )
        )
    )
    rr.log("checkpoints", rr.Arrows2D(vectors=arrows[:, 2:], origins=arrows[:, :2]))

    max_c = 0.1  # max curvature
    path_x, path_y, yaw, directions = [], [], [], []

    for i in range(len(states) - 1):
        s_x = states[i][0]
        s_y = states[i][1]
        s_yaw = radians(states[i][2])
        g_x = states[i + 1][0]
        g_y = states[i + 1][1]
        g_yaw = radians(states[i + 1][2])

        path_i = calc_optimal_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c)

        path_x += path_i.x
        path_y += path_i.y
        yaw += path_i.yaw
        directions += path_i.directions
        rr.log("path", rr.LineStrips2D(np.array([path_x, path_y]).T))

    car = Car(
        4.5,
        1.0,
        3.0,
        0.7 * 3.0,
        3.5,
        0.5,
        1.0,
    )

    for k in range(len(path_x)):
        if k < len(path_x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / 0.4
            steer = pi_2_pi(atan2(-car.WB * dy, directions[k]))
        else:
            steer = 0.0

        draw_car(path_x[k], path_y[k], yaw[k], steer, car)


if __name__ == "__main__":
    main()
