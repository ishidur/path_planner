"""
Hybrid A*
@author: Huiming Zhou
https://github.com/zhm-real/MotionPlanning
with some minor modifications made by Ryota Ishidu
"""

import math
from dataclasses import dataclass

import numpy as np
import rerun as rr
from heapdict import heapdict
from scipy.spatial import KDTree

from .astar import calc_holonomic_heuristic_with_obstacle
from .reeds_shepp import Car, RSPath, calc_all_paths, pi_2_pi


@dataclass
class HybridAstarConfig:  # Parameter config
    XY_RESO: float  # [m]
    YAW_RESO: float  # [rad]
    MOVE_STEP: float  # [m] path interporate resolution
    N_STEER: int  # steer command number
    COLLISION_CHECK_STEP: int  # skip number for collision check
    EXTEND_BOUND: int  # collision check range extended

    GEAR_COST: float  # switch back penalty cost
    BACKWARD_COST: float  # backward penalty cost
    STEER_CHANGE_COST: float  # steer angle change penalty cost
    STEER_ANGLE_COST: float  # steer angle penalty cost
    H_COST: float  # Heuristic cost penalty cost
    car: Car
    MAX_STEER: float  # [rad] maximum steering angle


@dataclass
class Node:
    xind: int
    yind: int
    yawind: int
    direction: int
    x: list[float]
    y: list[float]
    yaw: list[float]
    directions: list[float]
    steer: float
    cost: float
    pind: int


@dataclass
class Param:
    minx: int
    miny: int
    minyaw: int
    maxx: int
    maxy: int
    xw: int
    yw: int
    xyreso: float
    yawreso: float
    ox: list[float]
    oy: list[float]
    kdtree: KDTree


@dataclass
class HybridAstarPath:
    x: list[float]
    y: list[float]
    yaw: list[float]
    direction: list[int]
    cost: float


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self) -> bool:
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority: float):
        self.queue[item] = priority  # push

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority


def hybrid_astar_planning(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    ox: list[int],
    oy: list[int],
    xyreso: float,
    yawreso: float,
    conf: HybridAstarConfig,
) -> HybridAstarPath | None:
    sxr, syr = int(np.round(sx / xyreso)), int(np.round(sy / xyreso))
    gxr, gyr = int(np.round(gx / xyreso)), int(np.round(gy / xyreso))
    syawr = int(np.round(pi_2_pi(syaw) / yawreso))
    gyawr = int(np.round(pi_2_pi(gyaw) / yawreso))

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set(conf)
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P, conf))
    explored = []

    while len(open_set) != 0:
        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)
        # explored.append([[n_curr.x[0], n_curr.y[0]], [n_curr.x[-1], n_curr.y[-1]]])
        explored.append(list(zip(n_curr.x, n_curr.y)))
        rr.log("explored", rr.LineStrips2D(explored))

        if fpath := update_node_with_analystic_expantion(n_curr, ngoal, P, conf):
            return extract_path(closed_set, fpath, nstart)

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P, conf)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P, conf))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P, conf))
    return None


def extract_path(closed: dict, ngoal: Node, nstart: Node) -> HybridAstarPath:
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while node.pind >= 0:
        rx += node.x[:1:-1]
        ry += node.y[:1:-1]
        ryaw += node.yaw[:1:-1]
        direc += node.directions[:1:-1]
        cost += node.cost

        node = closed[node.pind]

    rx += nstart.x[::-1]
    ry += nstart.y[::-1]
    ryaw += nstart.yaw[::-1]
    direc += nstart.directions[::-1]
    cost += nstart.cost

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = HybridAstarPath(rx, ry, ryaw, direc, cost)

    return path


def calc_next_node(
    n_curr: Node, c_id: int, u: float, d: float, P: Param, conf: HybridAstarConfig
) -> Node | None:
    step = conf.XY_RESO * 2

    nlist = math.ceil(step / conf.MOVE_STEP)
    xlist = [n_curr.x[-1]] * (nlist + 1)
    ylist = [n_curr.y[-1]] * (nlist + 1)
    yawlist = [n_curr.yaw[-1]] * (nlist + 1)

    for i in range(nlist):
        xlist[i + 1] = xlist[i] + d * conf.MOVE_STEP * math.cos(yawlist[i])
        ylist[i + 1] = ylist[i] + d * conf.MOVE_STEP * math.sin(yawlist[i])
        yawlist[i + 1] = pi_2_pi(
            yawlist[i] + d * conf.MOVE_STEP / conf.car.wheel_base * math.tan(u)
        )

    xind = int(np.round(xlist[-1] / P.xyreso))
    yind = int(np.round(ylist[-1] / P.xyreso))
    yawind = int(np.round(yawlist[-1] / P.yawreso))

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P, conf):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * conf.BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += conf.GEAR_COST

    cost += conf.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    cost += conf.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(
        xind, yind, yawind, direction, xlist, ylist, yawlist, directions, u, cost, c_id
    )

    return node


def is_index_ok(
    xind: int,
    yind: int,
    xlist: list[float],
    ylist: list[float],
    yawlist: list[float],
    P: Param,
    conf: HybridAstarConfig,
) -> bool:
    if xind <= P.minx or xind >= P.maxx or yind <= P.miny or yind >= P.maxy:
        return False

    ind = range(0, len(xlist), conf.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    return not is_collision(nodex, nodey, nodeyaw, P, conf)


def update_node_with_analystic_expantion(
    n_curr: Node, ngoal: Node, P: Param, conf: HybridAstarConfig
) -> Node | None:
    path = analystic_expantion(n_curr, ngoal, P, conf)  # rs path: n -> ngoal

    if not path:
        return None

    fx = path.traj_x
    fy = path.traj_y
    fyaw = path.traj_yaw
    fd = path.traj_dirs

    fcost = n_curr.cost + calc_rs_path_cost(path, conf)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(
        n_curr.xind,
        n_curr.yind,
        n_curr.yawind,
        n_curr.direction,
        fx,
        fy,
        fyaw,
        fd,
        fsteer,
        fcost,
        fpind,
    )

    return fpath


def analystic_expantion(
    node: Node, ngoal: Node, P: Param, conf: HybridAstarConfig
) -> RSPath | None:
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(conf.MAX_STEER) / conf.car.wheel_base
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=conf.MOVE_STEP)

    if paths is None:
        return None

    pq = QueuePrior()
    for path in paths:
        pq.put(path, calc_rs_path_cost(path, conf))

    while not pq.empty():
        path = pq.get()
        ind = range(0, len(path.traj_x), conf.COLLISION_CHECK_STEP)

        pathx = [path.traj_x[k] for k in ind]
        pathy = [path.traj_y[k] for k in ind]
        pathyaw = [path.traj_yaw[k] for k in ind]

        if not is_collision(pathx, pathy, pathyaw, P, conf):
            return path

    return None


def is_collision(
    x: list[float], y: list[float], yaw: list[float], P: Param, conf: HybridAstarConfig
) -> bool:
    cp2center = (conf.car.center2front - conf.car.center2back) / 2.0
    bound_len = (conf.car.center2front + conf.car.center2back) / 2.0 + conf.EXTEND_BOUND
    bound_wid = conf.car.width / 2 + conf.EXTEND_BOUND
    bound_radius = math.hypot(bound_len, bound_wid)
    for ix, iy, iyaw in zip(x, y, yaw):
        cx = ix + cp2center * math.cos(iyaw)
        cy = iy + cp2center * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], bound_radius)

        if not ids:
            continue

        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < bound_len and abs(dy) < bound_wid:
                return True

    return False


def calc_rs_path_cost(rspath: RSPath, conf: HybridAstarConfig) -> float:
    cost = 0.0

    for lr in rspath.move_length:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * conf.BACKWARD_COST

    for i in range(len(rspath.move_length) - 1):
        if rspath.move_length[i] * rspath.move_length[i + 1] < 0.0:
            cost += conf.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += conf.STEER_ANGLE_COST * abs(conf.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -conf.MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = conf.MAX_STEER

    for i in range(nctypes - 1):
        cost += conf.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def calc_hybrid_cost(
    node: Node, hmap: list[list[float]], P: Param, conf: HybridAstarConfig
) -> float:
    cost = node.cost + conf.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_motion_set(conf: HybridAstarConfig) -> tuple[list[float], list[float]]:
    s = np.arange(
        conf.MAX_STEER / conf.N_STEER,
        conf.MAX_STEER,
        conf.MAX_STEER / conf.N_STEER,
    )

    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def calc_index(node: Node, P: Param) -> int:
    ind = (
        (node.yawind - P.minyaw) * P.xw * P.yw
        + (node.yind - P.miny) * P.xw
        + (node.xind - P.minx)
    )

    return ind


def calc_parameters(
    ox: list[int], oy: list[int], xyreso: float, yawreso: float, kdtree: KDTree
) -> Param:
    minx = int(np.round(min(ox) / xyreso))
    miny = int(np.round(min(oy) / xyreso))
    maxx = int(np.round(max(ox) / xyreso))
    maxy = int(np.round(max(oy) / xyreso))

    xw, yw = maxx - minx, maxy - miny

    minyaw = int(np.round(-np.pi / yawreso)) - 1

    return Param(
        minx,
        miny,
        minyaw,
        maxx,
        maxy,
        xw,
        yw,
        xyreso,
        yawreso,
        ox,
        oy,
        kdtree,
    )
