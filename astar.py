# https://github.com/zhm-real/MotionPlanning

import heapq
import math
import numpy as np
from dataclasses import dataclass
import rerun as rr


@dataclass
class HolonomicNode:
    x: int  # x position of node
    y: int  # y position of node
    cost: float  # g cost of node
    pind: int  # parent index of node


@dataclass
class Param:
    minx: int
    miny: int
    maxx: int
    maxy: int
    xw: int
    yw: int
    reso: float  # resolution of grid world
    motion: list[tuple[int, int]]  # motion set


def astar_planning(
    sx: float,
    sy: float,
    gx: float,
    gy: float,
    ox: list[float],
    oy: list[float],
    reso: float,
    robo_radius: float,
) -> tuple[list[float], list[float]] | None:
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """

    n_start = HolonomicNode(int(np.round(sx / reso)), int(np.round(sy / reso)), 0.0, -1)
    n_goal = HolonomicNode(int(np.round(gx / reso)), int(np.round(gy / reso)), 0.0, -1)
    ox = [x / reso for x in ox]  # in grid, float
    oy = [y / reso for y in oy]  # in grid, float

    param, obsmap = calc_parameters(ox, oy, robo_radius, reso)
    goal_ind = calc_index(n_goal, param)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, param)] = n_start

    q_priority = []
    heapq.heappush(q_priority, (fvalue(n_start, n_goal), calc_index(n_start, param)))

    while len(open_set) != 0:
        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)
        if goal_ind == ind:
            exps = []
            for n in closed_set.values():
                exps.append([n.x, n.y])
            rr.log("explored", rr.Points2D(np.array(exps)))
            pathx, pathy = extract_path(closed_set, n_start, n_goal, param)
            return pathx, pathy

        for i in range(len(param.motion)):
            node = HolonomicNode(
                n_curr.x + param.motion[i][0],
                n_curr.y + param.motion[i][1],
                n_curr.cost + u_cost(param.motion[i]),
                ind,
            )

            if not check_node(node, param, obsmap):
                continue

            n_ind = calc_index(node, param)

            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(
                        q_priority, (fvalue(node, n_goal), calc_index(node, param))
                    )

    exps = []
    for n in closed_set.values():
        exps.append([n.x, n.y])
    rr.log("explored", rr.Points2D(np.array(exps)))
    return None


def calc_holonomic_heuristic_with_obstacle(
    node, ox: list[float], oy: list[float], reso: float, rr: float
) -> list[list[float]]:
    n_goal = HolonomicNode(
        int(np.round(node.x[-1] / reso)), int(np.round(node.y[-1] / reso)), 0.0, -1
    )

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))

    while open_set:
        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for mot in P.motion:
            node = HolonomicNode(
                n_curr.x + mot[0],
                n_curr.y + mot[1],
                n_curr.cost + u_cost(mot),
                ind,
            )

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap


def check_node(node: HolonomicNode, P: Param, obsmap: list[list[bool]]) -> bool:
    if node.x <= P.minx or node.x >= P.maxx or node.y <= P.miny or node.y >= P.maxy:
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False

    return True


def u_cost(u: tuple[int, int]) -> float:
    return math.hypot(u[0], u[1])


def fvalue(node: HolonomicNode, n_goal: HolonomicNode) -> float:
    return node.cost + h(node, n_goal)


def h(node: HolonomicNode, n_goal: HolonomicNode) -> float:
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)


def calc_index(node: HolonomicNode, P: Param) -> int:
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


def calc_parameters(
    ox: list[float], oy: list[float], rr: float, reso: float
) -> tuple[Param, list[list[bool]]]:
    minx, miny = int(np.round(min(ox))), int(np.round(min(oy)))
    maxx, maxy = int(np.round(max(ox))), int(np.round(max(oy)))
    xw, yw = maxx - minx, maxy - miny

    motion = get_motion()
    P = Param(minx, miny, maxx, maxy, xw, yw, reso, motion)
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap


def calc_obsmap(
    ox: list[float], oy: list[float], rr: float, P: Param
) -> list[list[bool]]:
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    return obsmap


def extract_path(
    closed_set: dict, n_start: dict, n_goal: HolonomicNode, P: Param
) -> tuple[list[float], list[float]]:
    pathx, pathy = [n_goal.x], [n_goal.y]
    n_ind = calc_index(n_goal, P)

    while True:
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        n_ind = node.pind

        if node == n_start:
            break

    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]

    return pathx, pathy


def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion


def get_env() -> tuple[list[int], list[int]]:
    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    return ox, oy


def main():
    rr.init("astar")
    rr.connect_tcp("172.28.64.1:9876")

    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    robot_radius = 2.0
    grid_resolution = 1.0
    ox, oy = get_env()
    rr.log("obstacles", rr.Points2D(np.array((ox, oy)).T))
    rr.log("start", rr.Points2D((sx, sy)))
    rr.log("goal", rr.Points2D((gx, gy)))

    res = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)
    if res is None:
        print("Searching failed!")
        return
    pathx, pathy = res
    rr.log("path", rr.LineStrips2D(np.array((pathx, pathy)).T))


if __name__ == "__main__":
    main()
