use crate::data_struct::Node;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f64;

use ordered_float::NotNan;
use rerun::external::glam::Vec2;

#[derive(Debug, Clone, Copy, PartialEq)]
struct HolonomicNode {
    x: usize,    // x position of node
    y: usize,    // y position of node
    cost: f64,   // g cost of node
    pind: isize, // parent index of node
}

#[derive(Debug, Clone)]
struct Param {
    minx: usize,
    miny: usize,
    maxx: usize,
    maxy: usize,
    xw: usize,
    yw: usize,
    reso: f64,                   // resolution of grid world
    motion: Vec<(isize, isize)>, // motion set
}

impl HolonomicNode {
    fn new(x: usize, y: usize, cost: f64, pind: isize) -> Self {
        HolonomicNode { x, y, cost, pind }
    }
}

impl Param {
    fn new(minx: usize, miny: usize, maxx: usize, maxy: usize, reso: f64) -> Self {
        let xw = maxx - minx;
        let yw = maxy - miny;
        let motion = get_motion();
        Param {
            minx,
            miny,
            maxx,
            maxy,
            xw,
            yw,
            reso,
            motion,
        }
    }
}

pub fn astar_planning(
    sx: f64,
    sy: f64,
    gx: f64,
    gy: f64,
    ox: &[f64],
    oy: &[f64],
    reso: f64,
    rr: f64,
    rec: &rerun::RecordingStream,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_start = HolonomicNode::new(
        (sx / reso).round() as usize,
        (sy / reso).round() as usize,
        0.0,
        -1,
    );
    let n_goal = HolonomicNode::new(
        (gx / reso).round() as usize,
        (gy / reso).round() as usize,
        0.0,
        -1,
    );

    let ox: Vec<f64> = ox.iter().map(|&x| x / reso).collect();
    let oy: Vec<f64> = oy.iter().map(|&y| y / reso).collect();

    let (param, obsmap) = calc_parameters(&ox, &oy, rr, reso);
    let goal_ind = calc_index(&n_goal, &param);
    let mut open_set: HashMap<usize, HolonomicNode> = HashMap::new();
    let mut closed_set: HashMap<usize, HolonomicNode> = HashMap::new();
    open_set.insert(calc_index(&n_start, &param), n_start);

    let mut q_priority = BinaryHeap::new();
    q_priority.push((
        Reverse(NotNan::new(fvalue(n_start, n_goal)).unwrap()),
        calc_index(&n_start, &param),
    ));
    let mut explored: Vec<Vec2> = Vec::new();

    while !open_set.is_empty() {
        let (_, ind) = q_priority.pop().unwrap();
        let n_curr = open_set.remove(&ind).unwrap();
        closed_set.insert(ind, n_curr);
        explored.push(Vec2::new(n_curr.x as f32, n_curr.y as f32));
        let _ = rec.log("explored", &rerun::Points2D::new(explored.clone()));
        if goal_ind == ind {
            return Some(extract_path(&closed_set, &n_start, &n_goal, &param));
        }

        for motion in &param.motion {
            let node = HolonomicNode::new(
                (n_curr.x as isize + motion.0) as usize,
                (n_curr.y as isize + motion.1) as usize,
                n_curr.cost + u_cost(*motion),
                ind as isize,
            );

            if !check_node(&node, &param, &obsmap) {
                continue;
            }

            let n_ind = calc_index(&node, &param);
            if !closed_set.contains_key(&n_ind) {
                if let Some(open_node) = open_set.get_mut(&n_ind) {
                    if open_node.cost > node.cost {
                        open_node.cost = node.cost;
                        open_node.pind = ind as isize;
                    }
                } else {
                    open_set.insert(n_ind, node);
                    q_priority.push((Reverse(NotNan::new(fvalue(node, n_goal)).unwrap()), n_ind));
                }
            }
        }
    }
    None
}

pub fn calc_holonomic_heuristic_with_obstacle(
    node: &Node,
    ox: &Vec<f64>,
    oy: &Vec<f64>,
    reso: f64,
    rr: f64,
) -> Vec<Vec<f64>> {
    let n_goal = HolonomicNode::new(
        (node.x.last().unwrap() / reso).round() as usize,
        (node.y.last().unwrap() / reso).round() as usize,
        0.0,
        -1,
    );

    let oxf: Vec<f64> = ox.iter().map(|&x| x / reso).collect();
    let oyf: Vec<f64> = oy.iter().map(|&x| x / reso).collect();

    let (param, obsmap) = calc_parameters(&oxf, &oyf, rr, reso);

    let mut open_set = HashMap::new();
    let mut closed_set = HashMap::new();
    open_set.insert(calc_index(&n_goal, &param), n_goal);

    let mut q_priority = BinaryHeap::new();
    q_priority.push((
        Reverse(NotNan::new(n_goal.cost).unwrap()),
        calc_index(&n_goal, &param),
    ));

    while !open_set.is_empty() {
        let (_, ind) = q_priority.pop().unwrap();
        let n_curr = open_set.remove(&ind).unwrap();
        closed_set.insert(ind, n_curr);
        for cmd in param.motion.clone() {
            let node = HolonomicNode::new(
                (n_curr.x as isize + cmd.0) as usize,
                (n_curr.y as isize + cmd.1) as usize,
                n_curr.cost + u_cost(cmd),
                ind as isize,
            );

            if !check_node(&node, &param, &obsmap) {
                continue;
            }
            let n_ind = calc_index(&node, &param);
            if !closed_set.contains_key(&n_ind) {
                if let Some(open_node) = open_set.get_mut(&n_ind) {
                    if open_node.cost > node.cost {
                        open_node.cost = node.cost;
                        open_node.pind = ind as isize;
                    }
                } else {
                    open_set.insert(n_ind, node);
                    q_priority.push((Reverse(NotNan::new(node.cost).unwrap()), n_ind));
                }
            }
        }
    }

    let mut hmap = vec![vec![f64::INFINITY; param.yw]; param.xw];
    closed_set.into_values().for_each(|n| {
        hmap[n.x - param.minx][n.y - param.miny] = n.cost;
    });
    hmap
}

fn check_node(node: &HolonomicNode, param: &Param, obsmap: &[Vec<bool>]) -> bool {
    if node.x <= param.minx || node.x >= param.maxx || node.y <= param.miny || node.y >= param.maxy
    {
        return false;
    }

    if obsmap[node.x - param.minx][node.y - param.miny] {
        return false;
    }

    true
}

fn u_cost(u: (isize, isize)) -> f64 {
    ((u.0 * u.0 + u.1 * u.1) as f64).sqrt()
}

fn fvalue(node: HolonomicNode, n_goal: HolonomicNode) -> f64 {
    node.cost + h(node, n_goal)
}

fn h(node: HolonomicNode, n_goal: HolonomicNode) -> f64 {
    (((node.x as isize - n_goal.x as isize).pow(2) + (node.y as isize - n_goal.y as isize).pow(2))
        as f64)
        .sqrt()
}

fn calc_index(node: &HolonomicNode, param: &Param) -> usize {
    (node.y - param.miny) * param.xw + (node.x - param.minx)
}

fn calc_parameters(ox: &Vec<f64>, oy: &Vec<f64>, rr: f64, reso: f64) -> (Param, Vec<Vec<bool>>) {
    let minx = (ox.iter().cloned().fold(f64::INFINITY, f64::min)).round() as usize;
    let miny = (oy.iter().cloned().fold(f64::INFINITY, f64::min)).round() as usize;
    let maxx = (ox.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).round() as usize;
    let maxy = (oy.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).round() as usize;

    let param = Param::new(minx, miny, maxx, maxy, reso);
    let obsmap = calc_obsmap(ox, oy, rr, &param);

    (param, obsmap)
}

fn calc_obsmap(ox: &Vec<f64>, oy: &Vec<f64>, rr: f64, param: &Param) -> Vec<Vec<bool>> {
    let mut obsmap = vec![vec![false; param.yw]; param.xw];

    for x in 0..param.xw {
        let xx = x + param.minx;
        for y in 0..param.yw {
            let yy = y + param.miny;
            for (&oxx, &oyy) in ox.iter().zip(oy.iter()) {
                if ((oxx - xx as f64).hypot(oyy - yy as f64)) <= rr / param.reso {
                    obsmap[x][y] = true;
                    break;
                }
            }
        }
    }

    obsmap
}

fn extract_path(
    closed_set: &HashMap<usize, HolonomicNode>,
    n_start: &HolonomicNode,
    n_goal: &HolonomicNode,
    param: &Param,
) -> (Vec<f64>, Vec<f64>) {
    let mut pathx = vec![n_goal.x as f64];
    let mut pathy = vec![n_goal.y as f64];
    let mut n_ind = calc_index(n_goal, param);

    while let Some(node) = closed_set.get(&n_ind) {
        pathx.push(node.x as f64);
        pathy.push(node.y as f64);
        n_ind = node.pind as usize;

        if node == n_start {
            break;
        }
    }

    pathx.reverse();
    pathy.reverse();

    for x in &mut pathx {
        *x *= param.reso;
    }

    for y in &mut pathy {
        *y *= param.reso;
    }

    (pathx, pathy)
}

fn get_motion() -> Vec<(isize, isize)> {
    vec![
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]
}

pub fn get_env() -> (Vec<f64>, Vec<f64>) {
    let mut ox = Vec::new();
    let mut oy = Vec::new();

    for i in 0..60 {
        ox.push(i as f64);
        oy.push(0.0);
    }
    for i in 0..60 {
        ox.push(60.0);
        oy.push(i as f64);
    }
    for i in 0..61 {
        ox.push(i as f64);
        oy.push(60.0);
    }
    for i in 0..61 {
        ox.push(0.0);
        oy.push(i as f64);
    }
    for i in 0..40 {
        ox.push(20.0);
        oy.push(i as f64);
    }
    for i in 0..40 {
        ox.push(40.0);
        oy.push(60.0 - i as f64);
    }

    (ox, oy)
}
