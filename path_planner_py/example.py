import time

import numpy as np
import rerun as rr

from path_planner_py import (
    Car,
    HybridAstarConfig,
    astar_planning,
    calc_optimal_path,
    draw_car,
    hybrid_astar_planning,
)


def astar():
    rr.init("astar", spawn=True)

    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    robot_radius = 2.0
    grid_resolution = 1.0

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
    rr.log("obstacles", rr.Points2D(np.array((ox, oy)).T))
    rr.log("start", rr.Points2D((sx, sy)))
    rr.log("goal", rr.Points2D((gx, gy)))

    res = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)
    if res is None:
        print("Searching failed!")
        return
    pathx, pathy = res
    rr.log("path", rr.LineStrips2D(np.array((pathx, pathy)).T))


def design_obstacles(x: int, y: int) -> tuple[list[int], list[int]]:
    ox, oy = [], []

    for i in range(x):
        ox.append(i)
        oy.append(0)
    for i in range(x):
        ox.append(i)
        oy.append(y - 1)
    for i in range(y):
        ox.append(0)
        oy.append(i)
    for i in range(y):
        ox.append(x - 1)
        oy.append(i)
    for i in range(10, 21):
        ox.append(i)
        oy.append(15)
    for i in range(15):
        ox.append(20)
        oy.append(i)
    for i in range(15, 30):
        ox.append(30)
        oy.append(i)
    for i in range(16):
        ox.append(40)
        oy.append(i)

    return ox, oy


def hyb_astar():
    rr.init("hybrid_astar", spawn=True)
    print("hybrid_astar start!")

    car = Car(
        4.5,
        1.0,
        3.0,
        3.5,
    )
    this_conf = HybridAstarConfig(
        2.0,
        np.deg2rad(15.0),
        0.4,
        20,
        5,
        1.4,
        100.0,
        5.0,
        5.0,
        1.0,
        15.0,
        car,
        0.6,
    )
    x, y = 51, 31
    sx, sy, syaw0 = 10.0, 7.0, np.deg2rad(120.0)
    gx, gy, gyaw0 = 45.0, 20.0, np.deg2rad(90.0)
    rr.log("start", rr.Points2D((sx, sy)))
    rr.log("goal", rr.Points2D((gx, gy)))

    ox, oy = design_obstacles(x, y)
    rr.log(
        "obstacles", rr.Boxes2D(sizes=[1, 1] * len(ox), centers=np.array((ox, oy)).T)
    )

    t0 = time.perf_counter()
    path = hybrid_astar_planning(
        sx,
        sy,
        syaw0,
        gx,
        gy,
        gyaw0,
        ox,
        oy,
        this_conf.XY_RESO,
        this_conf.YAW_RESO,
        this_conf,
    )
    t1 = time.perf_counter()
    print("running T: ", t1 - t0)

    if not path:
        print("Searching failed!")
        return

    x = path.x
    y = path.y
    yaw = path.yaw
    rr.log("path", rr.LineStrips2D(np.array((x, y)).T))

    for k in range(len(x)):
        draw_car(x[k], y[k], yaw[k], car)

    print("Done!")


def reeds_shepp():
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
                    np.cos(np.deg2rad(s[2])),
                    np.sin(np.deg2rad(s[2])),
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
        s_yaw = np.deg2rad(states[i][2])
        g_x = states[i + 1][0]
        g_y = states[i + 1][1]
        g_yaw = np.deg2rad(states[i + 1][2])

        path_i = calc_optimal_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c)

        path_x += path_i.traj_x
        path_y += path_i.traj_y
        yaw += path_i.traj_yaw
        directions += path_i.traj_dirs
        rr.log("path", rr.LineStrips2D(np.array([path_x, path_y]).T))

    car = Car(
        4.5,
        1.0,
        3.0,
        3.5,
    )

    for k in range(len(path_x)):
        draw_car(path_x[k], path_y[k], yaw[k], car)


reeds_shepp()
astar()
hyb_astar()
