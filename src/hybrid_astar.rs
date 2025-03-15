use crate::astar::calc_holonomic_heuristic_with_obstacle;
use crate::data_struct::Node;
use crate::reeds_shepp::{Car, RSPath, calc_all_paths, pi_2_pi};
use crate::util::rot2d;
use kdtree::KdTree;
use ordered_float::NotNan;
use rerun::external::glam::Vec2;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f64;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    xy_reso: f64,                // [m]
    yaw_reso: f64,               // [rad]
    pub move_step: f64,          // [m] path interpolate resolution
    n_steer: usize,              // steer command number
    collision_check_step: usize, // skip number for collision check
    extend_bound: f64,           // collision check range extended
    gear_cost: f64,              // switch back penalty cost
    backward_cost: f64,          // backward penalty cost
    steer_change_cost: f64,      // steer angle change penalty cost
    steer_angle_cost: f64,       // steer angle penalty cost
    h_cost: f64,                 // Heuristic cost penalty cost
    pub car: Car,
    max_steer: f64, // [rad] maximum steering angle
}

impl Config {
    pub fn new(
        xy_reso: f64,
        yaw_reso: f64,
        move_step: f64,
        n_steer: usize,
        collision_check_step: usize,
        extend_bound: f64,
        gear_cost: f64,
        backward_cost: f64,
        steer_change_cost: f64,
        steer_angle_cost: f64,
        h_cost: f64,
        car: Car,
        max_steer: f64,
    ) -> Self {
        Config {
            xy_reso,
            yaw_reso,
            move_step,
            n_steer,
            collision_check_step,
            extend_bound,
            gear_cost,
            backward_cost,
            steer_change_cost,
            steer_angle_cost,
            h_cost,
            car,
            max_steer,
        }
    }
}

#[derive(Debug, Clone)]
struct Param {
    minx: usize,
    miny: usize,
    minyaw: isize,
    maxx: usize,
    maxy: usize,
    xw: usize,
    yw: usize,
    xyreso: f64,
    yawreso: f64,
    ox: Vec<f64>,
    oy: Vec<f64>,
    kdtree: KdTree<f64, usize, [f64; 2]>,
}

#[derive(Debug)]
pub struct HybridAstarPath {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub yaw: Vec<f64>,
    pub direction: Vec<isize>,
    pub cost: f64,
}

pub fn hybrid_astar_planning(
    sx: f64,
    sy: f64,
    syaw: f64,
    gx: f64,
    gy: f64,
    gyaw: f64,
    ox: &Vec<f64>,
    oy: &Vec<f64>,
    conf: &Config,
    rec: Option<&rerun::RecordingStream>,
) -> Option<HybridAstarPath> {
    let xyreso = conf.xy_reso;
    let yawreso = conf.yaw_reso;
    let sxr = (sx / xyreso).round() as usize;
    let syr = (sy / xyreso).round() as usize;
    let gxr = (gx / xyreso).round() as usize;
    let gyr = (gy / xyreso).round() as usize;
    let syawr = (pi_2_pi(syaw) / yawreso).round() as isize;
    let gyawr = (pi_2_pi(gyaw) / yawreso).round() as isize;

    let nstart = Node::new(
        sxr,
        syr,
        syawr,
        1,
        vec![sx],
        vec![sy],
        vec![syaw],
        vec![1],
        0.0,
        0.0,
        -1,
    );
    let ngoal = Node::new(
        gxr,
        gyr,
        gyawr,
        1,
        vec![gx],
        vec![gy],
        vec![gyaw],
        vec![1],
        0.0,
        0.0,
        -1,
    );
    let mut kdtree = KdTree::new(2);
    for (i, (x, y)) in ox.iter().zip(oy.iter()).enumerate() {
        kdtree.add([*x as f64, *y as f64], i).unwrap();
    }

    let param = calc_parameters(ox, oy, xyreso, yawreso, kdtree);
    let hmap =
        calc_holonomic_heuristic_with_obstacle(&ngoal, &param.ox, &param.oy, param.xyreso, 1.0);
    let (steer_set, direc_set) = calc_motion_set(conf);
    let mut open_set: HashMap<usize, Node> = HashMap::new();
    let mut closed_set: HashMap<usize, Node> = HashMap::new();
    let st_idx = calc_index(&nstart, &param);
    open_set.insert(st_idx, nstart.clone());

    let mut prior_queue = BinaryHeap::new();
    prior_queue.push((
        Reverse(NotNan::new(calc_hybrid_cost(&nstart, &hmap, &param, conf)).unwrap()),
        st_idx,
    ));
    let mut explored: Vec<[Vec2; 2]> = Vec::new();

    while !open_set.is_empty() {
        let (_, ind) = prior_queue.pop().unwrap();

        if let Some(n_curr) = open_set.remove(&ind) {
            closed_set.insert(ind, n_curr.clone());
            if rec.is_some() {
                explored.push([
                    Vec2::new(
                        *n_curr.x.first().unwrap() as f32,
                        *n_curr.y.first().unwrap() as f32,
                    ),
                    Vec2::new(
                        *n_curr.x.last().unwrap() as f32,
                        *n_curr.y.last().unwrap() as f32,
                    ),
                ]);
                let _ = rec?.log("explored", &rerun::LineStrips2D::new(explored.clone()));
            }

            if let Some(fpath) = update_node_with_analystic_expansion(&n_curr, &ngoal, &param, conf)
            {
                return Some(extract_path(&closed_set, &fpath, &nstart));
            }
            for i in 0..steer_set.len() {
                if let Some(node) =
                    calc_next_node(&n_curr, ind, steer_set[i], direc_set[i], &param, conf)
                {
                    let node_ind = calc_index(&node, &param);

                    if closed_set.contains_key(&node_ind) {
                        continue;
                    }

                    if !open_set.contains_key(&node_ind) {
                        prior_queue.push((
                            Reverse(
                                NotNan::new(calc_hybrid_cost(&node, &hmap, &param, conf)).unwrap(),
                            ),
                            node_ind,
                        ));
                        open_set.insert(node_ind, node);
                    } else {
                        if open_set[&node_ind].cost > node.cost {
                            prior_queue.push((
                                Reverse(
                                    NotNan::new(calc_hybrid_cost(&node, &hmap, &param, conf))
                                        .unwrap(),
                                ),
                                node_ind,
                            ));
                            open_set.insert(node_ind, node);
                        }
                    }
                }
            }
        }
    }

    None
}

fn extract_path(closed: &HashMap<usize, Node>, ngoal: &Node, nstart: &Node) -> HybridAstarPath {
    let mut rx = vec![];
    let mut ry = vec![];
    let mut ryaw = vec![];
    let mut direc = vec![];
    let mut cost = 0.0;
    let mut node = ngoal.clone();

    while node.pind >= 0 {
        rx.extend(node.x[1..].iter().rev());
        ry.extend(node.y[1..].iter().rev());
        ryaw.extend(node.yaw[1..].iter().rev());
        direc.extend(node.directions[1..].iter().rev());
        cost += node.cost;
        node = closed[&(node.pind as usize)].clone();
    }

    rx.extend(nstart.x.iter().rev());
    ry.extend(nstart.y.iter().rev());
    ryaw.extend(nstart.yaw.iter().rev());
    direc.extend(nstart.directions.iter().rev());
    cost += nstart.cost;

    let path = HybridAstarPath {
        x: rx.into_iter().rev().collect(),
        y: ry.into_iter().rev().collect(),
        yaw: ryaw.into_iter().rev().collect(),
        direction: direc.into_iter().rev().collect(),
        cost,
    };

    path
}

fn calc_next_node(
    n_curr: &Node,
    c_id: usize,
    u: f64,
    d: f64,
    param: &Param,
    conf: &Config,
) -> Option<Node> {
    let step = conf.xy_reso * 2.0;

    let nlist = (step / conf.move_step).ceil() as usize;
    let &curr_x = n_curr.x.last().unwrap();
    let &curr_y = n_curr.y.last().unwrap();
    let &curr_yaw = n_curr.yaw.last().unwrap();
    let mut xlist = vec![curr_x; nlist + 1];
    let mut ylist = vec![curr_y; nlist + 1];
    let mut yawlist = vec![curr_yaw; nlist + 1];

    for i in 0..nlist {
        xlist[i + 1] = xlist[i] + d * conf.move_step * yawlist[i].cos();
        ylist[i + 1] = ylist[i] + d * conf.move_step * yawlist[i].sin();
        yawlist[i + 1] = yawlist[i] + d * conf.move_step / conf.car.wheel_base * u.tan();
    }

    let xind = (xlist.last().unwrap() / param.xyreso).round() as usize;
    let yind = (ylist.last().unwrap() / param.xyreso).round() as usize;
    let yawind = (yawlist.last().unwrap() / param.yawreso).round() as isize;

    if !is_index_ok(xind, yind, &xlist, &ylist, &yawlist, param, conf) {
        return None;
    }

    let mut cost = 0.0;

    let direction;
    if d > 0.0 {
        direction = 1;
        cost += step.abs();
    } else {
        direction = -1;
        cost += step.abs() * conf.backward_cost;
    }

    if direction != n_curr.direction {
        cost += conf.gear_cost;
    }

    cost += conf.steer_angle_cost * u.abs();
    cost += conf.steer_change_cost * (n_curr.steer - u).abs();
    cost += n_curr.cost;

    let directions = vec![direction; xlist.len()];

    let node = Node::new(
        xind,
        yind,
        yawind,
        direction,
        xlist,
        ylist,
        yawlist,
        directions,
        u,
        cost,
        c_id as isize,
    );

    Some(node)
}

fn is_index_ok(
    xind: usize,
    yind: usize,
    xlist: &[f64],
    ylist: &[f64],
    yawlist: &[f64],
    param: &Param,
    conf: &Config,
) -> bool {
    if xind <= param.minx || xind >= param.maxx || yind <= param.miny || yind >= param.maxy {
        return false;
    }

    let ind: Vec<usize> = (0..xlist.len())
        .step_by(conf.collision_check_step)
        .collect();
    let x_picked: Vec<f64> = ind.iter().map(|&k| xlist[k]).collect();
    let y_picked: Vec<f64> = ind.iter().map(|&k| ylist[k]).collect();
    let yaw_picked: Vec<f64> = ind.iter().map(|&k| yawlist[k]).collect();

    !is_collision(&x_picked, &y_picked, &yaw_picked, param, conf)
}

fn update_node_with_analystic_expansion(
    n_curr: &Node,
    ngoal: &Node,
    param: &Param,
    conf: &Config,
) -> Option<Node> {
    if let Some(path) = analystic_expansion(n_curr, ngoal, param, conf) {
        let fx = path.traj_x.to_vec();
        let fy = path.traj_y.to_vec();
        let fyaw = path.traj_yaw.to_vec();
        let fd = path.traj_dirs.to_vec();

        let fcost = n_curr.cost + calc_rs_path_cost(&path, conf);
        let fpind = calc_index(n_curr, param);
        let fsteer = 0.0;

        let fpath = Node::new(
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
            fpind as isize,
        );

        return Some(fpath);
    }
    None
}

fn analystic_expansion(node: &Node, ngoal: &Node, param: &Param, conf: &Config) -> Option<RSPath> {
    let sx = node.x.last().unwrap();
    let sy = node.y.last().unwrap();
    let syaw = node.yaw.last().unwrap();
    let gx = ngoal.x.last().unwrap();
    let gy = ngoal.y.last().unwrap();
    let gyaw = ngoal.yaw.last().unwrap();

    let maxc = conf.max_steer.tan() / conf.car.wheel_base;
    let paths = calc_all_paths(*sx, *sy, *syaw, *gx, *gy, *gyaw, maxc, conf.move_step);

    let mut pq = BinaryHeap::new();
    for i in 0..paths.len() {
        pq.push((
            Reverse(NotNan::new(calc_rs_path_cost(&paths[i], conf)).unwrap()),
            i,
        ));
    }

    while !pq.is_empty() {
        let (_, path_idx) = pq.pop().unwrap();
        let ind: Vec<usize> = (0..paths[path_idx].traj_x.len())
            .step_by(conf.collision_check_step)
            .collect();

        let pathx: Vec<f64> = ind.iter().map(|&k| paths[path_idx].traj_x[k]).collect();
        let pathy: Vec<f64> = ind.iter().map(|&k| paths[path_idx].traj_y[k]).collect();
        let pathyaw: Vec<f64> = ind.iter().map(|&k| paths[path_idx].traj_yaw[k]).collect();

        if !is_collision(&pathx, &pathy, &pathyaw, param, conf) {
            return Some(paths[path_idx].clone());
        }
    }

    None
}

fn is_collision(x: &Vec<f64>, y: &Vec<f64>, yaw: &Vec<f64>, param: &Param, conf: &Config) -> bool {
    let cp2center = (conf.car.center2front - conf.car.center2back) / 2.0;
    let bound_len = (conf.car.center2front + conf.car.center2back) / 2.0 + conf.extend_bound;
    let bound_wid = conf.car.width / 2.0 + conf.extend_bound;
    let bound_radius = bound_len.hypot(bound_wid);
    for ((&ix, &iy), &iyaw) in x.iter().zip(y.iter()).zip(yaw.iter()) {
        let cx = ix + cp2center * iyaw.cos();
        let cy = iy + cp2center * iyaw.sin();

        let ids = param
            .kdtree
            .within(&[cx, cy], bound_radius, &|a, b| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .unwrap();
        if ids.is_empty() {
            continue;
        }

        for (_, &i) in ids {
            let xo = param.ox[i] - cx;
            let yo = param.oy[i] - cy;
            let (dx, dy) = rot2d(&[xo, yo], -iyaw);

            if (dx.abs() < bound_len) && (dy.abs() < bound_wid) {
                return true;
            }
        }
    }

    false
}

fn calc_rs_path_cost(rspath: &RSPath, conf: &Config) -> f64 {
    let mut cost = 0.0;

    for &lr in &rspath.lengths {
        if lr >= 0.0 {
            cost += 1.0;
        } else {
            cost += lr.abs() * conf.backward_cost;
        }
    }

    for i in 0..rspath.lengths.len() - 1 {
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0 {
            cost += conf.gear_cost;
        }
    }

    for &ctype in &rspath.ctypes {
        if ctype != 'S' {
            cost += conf.steer_angle_cost * conf.max_steer.abs();
        }
    }

    let nctypes = rspath.ctypes.len();
    let mut ulist = vec![0.0; nctypes];

    for i in 0..nctypes {
        if rspath.ctypes[i] == 'R' {
            ulist[i] = -conf.max_steer;
        } else if rspath.ctypes[i] == 'L' {
            ulist[i] = conf.max_steer;
        }
    }

    for i in 0..nctypes - 1 {
        cost += conf.steer_change_cost * (ulist[i + 1] - ulist[i]).abs();
    }

    cost
}

fn calc_hybrid_cost(node: &Node, hmap: &Vec<Vec<f64>>, param: &Param, conf: &Config) -> f64 {
    node.cost + conf.h_cost * hmap[node.xind - param.minx][node.yind - param.miny]
}

fn calc_motion_set(conf: &Config) -> (Vec<f64>, Vec<f64>) {
    let s: Vec<f64> = (1..conf.n_steer as usize)
        .map(|i| (i as f64) * conf.max_steer / conf.n_steer as f64)
        .collect();
    let mut steer: Vec<f64> = s
        .iter()
        .cloned()
        .chain(vec![0.0])
        .chain(s.iter().map(|&v| -v))
        .collect();
    let direc: Vec<f64> = steer
        .iter()
        .map(|_| 1.)
        .chain(steer.iter().map(|_| -1.))
        .collect();

    steer.extend(steer.clone());

    (steer, direc)
}

fn calc_index(node: &Node, param: &Param) -> usize {
    (node.yawind - param.minyaw) as usize * param.xw * param.yw
        + (node.yind - param.miny) * param.xw
        + (node.xind - param.minx)
}

fn calc_parameters(
    ox: &Vec<f64>,
    oy: &Vec<f64>,
    xyreso: f64,
    yawreso: f64,
    kdtree: KdTree<f64, usize, [f64; 2]>,
) -> Param {
    let minx = (ox.iter().cloned().fold(f64::INFINITY, f64::min) as f64 / xyreso).round() as usize;
    let miny = (oy.iter().cloned().fold(f64::INFINITY, f64::min) as f64 / xyreso).round() as usize;
    let maxx =
        (ox.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as f64 / xyreso).round() as usize;
    let maxy =
        (oy.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as f64 / xyreso).round() as usize;

    let xw = maxx - minx;
    let yw = maxy - miny;

    let minyaw = (-PI / yawreso).round() as isize - 1;

    Param {
        minx,
        miny,
        minyaw,
        maxx,
        maxy,
        xw,
        yw,
        xyreso,
        yawreso,
        ox: ox.to_vec(),
        oy: oy.to_vec(),
        kdtree,
    }
}
