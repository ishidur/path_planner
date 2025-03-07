use std::{f64::consts::PI, vec};
const MAX_LENGTH: f64 = 1000.0;

#[derive(Clone, Debug)]
pub struct RSPath {
    pub lengths: Vec<f64>, // lengths of each part of path (+: forward, -: backward)
    pub ctypes: Vec<char>, // type of each part of the path
    pub total_length: f64, // total path length
    pub x: Vec<f64>,       // final x positions [m]
    pub y: Vec<f64>,       // final y positions [m]
    pub yaw: Vec<f64>,     // final yaw angles [rad]
    pub directions: Vec<isize>, // forward: 1, backward: -1
}

impl RSPath {
    fn new(
        lengths: Vec<f64>,
        ctypes: Vec<char>,
        total_length: f64,
        x: Vec<f64>,
        y: Vec<f64>,
        yaw: Vec<f64>,
        directions: Vec<isize>,
    ) -> Self {
        RSPath {
            lengths,
            ctypes,
            total_length,
            x,
            y,
            yaw,
            directions,
        }
    }
}
pub fn calc_optimal_path(
    sx: f64,
    sy: f64,
    syaw: f64,
    gx: f64,
    gy: f64,
    gyaw: f64,
    maxc: f64,
    step_size: f64,
) -> RSPath {
    let paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size);

    let (_, mini) =
        paths
            .iter()
            .enumerate()
            .fold((paths[0].total_length, 0), |(min_l, mini), (i, path)| {
                if path.total_length <= min_l {
                    (path.total_length, i)
                } else {
                    (min_l, mini)
                }
            });

    paths[mini].clone()
}

pub fn calc_all_paths(
    sx: f64,
    sy: f64,
    syaw: f64,
    gx: f64,
    gy: f64,
    gyaw: f64,
    maxc: f64,
    step_size: f64,
) -> Vec<RSPath> {
    let q0 = [sx, sy, syaw];
    let q1 = [gx, gy, gyaw];

    let mut paths = generate_path(&q0, &q1, maxc);

    for path in &mut paths {
        let (x, y, yaw, directions) = generate_local_course(
            path.total_length,
            &path.lengths,
            &path.ctypes,
            maxc,
            step_size * maxc,
        );

        // convert global coordinate
        path.x = x
            .iter()
            .zip(&y)
            .map(|(&ix, &iy)| ix * (-q0[2]).cos() + iy * (-q0[2]).sin() + q0[0])
            .collect();
        path.y = x
            .iter()
            .zip(&y)
            .map(|(&ix, &iy)| -ix * (-q0[2]).sin() + iy * (-q0[2]).cos() + q0[1])
            .collect();
        path.yaw = yaw.iter().map(|iyaw| pi_2_pi(iyaw + q0[2])).collect();
        path.directions = directions;
        path.lengths = path.lengths.iter().map(|l| l / maxc).collect();
        path.total_length /= maxc;
    }
    paths
}

fn set_path(paths: &mut Vec<RSPath>, lengths: Vec<f64>, ctypes: Vec<char>) {
    let mut path = RSPath::new(vec![], vec![], 0.0, vec![], vec![], vec![], vec![]);
    path.ctypes = ctypes;
    path.lengths = lengths.clone();

    // check same path exists
    for path_e in &mut *paths {
        if path_e.ctypes == path.ctypes {
            if path_e
                .lengths
                .iter()
                .zip(&path.lengths)
                .map(|(x, y)| x - y)
                .sum::<f64>()
                <= 0.01
            {
                return; // not insert path
            }
        }
    }

    path.total_length = path.lengths.iter().map(|l| l.abs()).sum();

    if path.total_length >= MAX_LENGTH {
        return;
    }

    assert!(path.total_length >= 0.01);
    paths.push(path);
}

fn lsl(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let (u, t) = r(x - phi.sin(), y - 1.0 + phi.cos());
    if t >= 0.0 {
        let v = m(phi - t);
        if v >= 0.0 {
            return (true, t, u, v);
        }
    }
    (false, 0., 0., 0.)
}

fn lsr(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let (mut u1, t1) = r(x + phi.sin(), y - 1.0 - phi.cos());
    u1 = u1.powi(2);

    if u1 >= 4. {
        let u = (u1 - 4.).sqrt();
        let theta = 2.0_f64.atan2(u);
        let t = m(t1 + theta);
        let v = m(t - phi);
        if (t >= 0.) & (v >= 0.0) {
            return (true, t, u, v);
        }
    }
    (false, 0., 0., 0.)
}

fn lrl(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let (u1, t1) = r(x - phi.sin(), y - 1.0 + phi.cos());

    if u1 <= 4. {
        let u = -2. * (0.25 * u1).asin();
        let t = m(t1 + 0.5 * u + PI);
        let v = m(phi - t + u);
        if (t >= 0.) & (u <= 0.0) {
            return (true, t, u, v);
        }
    }
    (false, 0., 0., 0.)
}

fn scs(x: f64, y: f64, phi: f64, paths: &mut Vec<RSPath>) {
    let (flag, t, u, v) = sls(x, y, phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['S', 'L', 'S']);
    }
    let (flag, t, u, v) = sls(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['S', 'R', 'S']);
    }
}

fn sls(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let _phi = m(phi);
    if (y > 0.) & (0. < _phi) & (_phi < PI * 0.99) {
        let xd = -y / (_phi).tan() + x;
        let t = xd - (_phi / 2.).tan();
        let u = _phi;
        let v = ((x - xd).powi(2) + y.powi(2)).sqrt() - (_phi / 2.).tan();
        return (true, t, u, v);
    } else if (y < 0.) & (0. < _phi) & (_phi < PI * 0.99) {
        let xd = -y / (_phi).tan() + x;
        let t = xd - (_phi / 2.).tan();
        let u = _phi;
        let v = -((x - xd).powi(2) + y.powi(2)).sqrt() - (_phi / 2.).tan();
        return (true, t, u, v);
    }
    (false, 0., 0., 0.)
}

fn csc(x: f64, y: f64, phi: f64, paths: &mut Vec<RSPath>) {
    let (flag, t, u, v) = lsl(x, y, phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['L', 'S', 'L']);
    }
    let (flag, t, u, v) = lsl(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, -u, -v], vec!['L', 'S', 'L']);
    }

    let (flag, t, u, v) = lsl(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['R', 'S', 'R']);
    }

    let (flag, t, u, v) = lsl(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, -u, -v], vec!['R', 'S', 'R']);
    }

    let (flag, t, u, v) = lsr(x, y, phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['L', 'S', 'R']);
    }

    let (flag, t, u, v) = lsr(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, -u, -v], vec!['L', 'S', 'R']);
    }

    let (flag, t, u, v) = lsr(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['R', 'S', 'L']);
    }

    let (flag, t, u, v) = lsr(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, -u, -v], vec!['R', 'S', 'L']);
    }
}

fn ccc(x: f64, y: f64, phi: f64, paths: &mut Vec<RSPath>) {
    let (flag, t, u, v) = lrl(x, y, phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrl(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, -u, -v], vec!['L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrl(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, u, v], vec!['R', 'L', 'R']);
    }

    let (flag, t, u, v) = lrl(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, -u, -v], vec!['R', 'L', 'R']);
    }

    // backwards
    let xb = x * phi.cos() + y * phi.sin();
    let yb = x * phi.sin() - y * phi.cos();

    let (flag, t, u, v) = lrl(xb, yb, phi);
    if flag {
        set_path(paths, vec![v, u, t], vec!['L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrl(-xb, yb, -phi);
    if flag {
        set_path(paths, vec![-v, -u, -t], vec!['L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrl(xb, -yb, -phi);
    if flag {
        set_path(paths, vec![v, u, t], vec!['R', 'L', 'R']);
    }

    let (flag, t, u, v) = lrl(-xb, -yb, phi);
    if flag {
        set_path(paths, vec![-v, -u, -t], vec!['R', 'L', 'R']);
    }
}

fn calc_tau_omega(u: f64, v: f64, xi: f64, eta: f64, phi: f64) -> (f64, f64) {
    let delta = m(u - v);
    let a = u.sin() - delta.sin();
    let b = u.cos() - delta.cos() - 1.0;
    let t1 = (eta * a - xi * b).atan2(xi * a + eta * b);
    let t2 = 2.0 * (delta.cos() - v.cos() - u.cos()) + 3.0;
    let tau;
    if t2 < 0. {
        tau = m(t1 + PI);
    } else {
        tau = m(t1);
    }
    let omega = m(tau - u + v - phi);
    (tau, omega)
}

fn lrlrn(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let rho = 0.25 * (2.0 + (xi.powi(2) + eta.powi(2)).sqrt());

    if rho <= 1.0 {
        let u = rho.cos();
        let (t, v) = calc_tau_omega(u, -u, xi, eta, phi);
        if (t >= 0.) & (v <= 0.) {
            return (true, t, u, v);
        }
    }
    (false, 0., 0., 0.)
}

fn lrlrp(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let rho = (20. - xi.powi(2) - eta.powi(2)) / 16.0;

    if (0.0 < rho) & (rho <= 1.0) {
        let u = -rho.cos();
        if u >= -0.5 * PI {
            let (t, v) = calc_tau_omega(u, u, xi, eta, phi);
            if (t >= 0.) & (v >= 0.) {
                return (true, t, u, v);
            }
        }
    }
    (false, 0., 0., 0.)
}

fn cccc(x: f64, y: f64, phi: f64, paths: &mut Vec<RSPath>) {
    let (flag, t, u, v) = lrlrn(x, y, phi);
    if flag {
        set_path(paths, vec![t, u, -u, v], vec!['L', 'R', 'L', 'R']);
    }

    let (flag, t, u, v) = lrlrn(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, -u, u, -v], vec!['L', 'R', 'L', 'R']);
    }

    let (flag, t, u, v) = lrlrn(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, u, -u, v], vec!['R', 'L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrlrn(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, -u, u, -v], vec!['R', 'L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrlrp(x, y, phi);
    if flag {
        set_path(paths, vec![t, u, u, v], vec!['L', 'R', 'L', 'R']);
    }

    let (flag, t, u, v) = lrlrp(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, -u, -u, -v], vec!['L', 'R', 'L', 'R']);
    }

    let (flag, t, u, v) = lrlrp(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, u, u, v], vec!['R', 'L', 'R', 'L']);
    }

    let (flag, t, u, v) = lrlrp(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, -u, -u, -v], vec!['R', 'L', 'R', 'L']);
    }
}

fn lrsr(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let (rho, theta) = r(-eta, xi);

    if rho >= 2.0 {
        let t = theta;
        let u = 2.0 - rho;
        let v = m(t + 0.5 * PI - phi);
        if (t >= 0.0) & (u <= 0.0) & (v <= 0.0) {
            return (true, t, u, v);
        }
    }
    (false, 0., 0., 0.)
}
fn lrsl(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let xi = x - phi.sin();
    let eta = y - 1.0 + phi.cos();
    let (rho, theta) = r(xi, eta);

    if rho >= 2.0 {
        let r = (rho.powi(2) - 4.0).sqrt();
        let u = 2.0 - r;
        let t = m(theta + (r.atan2(-2.0)));
        let v = m(phi - 0.5 * PI - t);
        if (t >= 0.0) & (u <= 0.0) & (v <= 0.0) {
            return (true, t, u, v);
        }
    }
    (false, 0., 0., 0.)
}

fn ccsc(x: f64, y: f64, phi: f64, paths: &mut Vec<RSPath>) {
    let (flag, t, u, v) = lrsl(x, y, phi);
    if flag {
        set_path(paths, vec![t, -0.5 * PI, u, v], vec!['L', 'R', 'S', 'L']);
    }

    let (flag, t, u, v) = lrsl(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, 0.5 * PI, -u, -v], vec!['L', 'R', 'S', 'L']);
    }

    let (flag, t, u, v) = lrsl(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, -0.5 * PI, u, v], vec!['R', 'L', 'S', 'R']);
    }

    let (flag, t, u, v) = lrsl(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, 0.5 * PI, -u, -v], vec!['R', 'L', 'S', 'R']);
    }

    let (flag, t, u, v) = lrsr(x, y, phi);
    if flag {
        set_path(paths, vec![t, -0.5 * PI, u, v], vec!['L', 'R', 'S', 'R']);
    }

    let (flag, t, u, v) = lrsr(-x, y, -phi);
    if flag {
        set_path(paths, vec![-t, 0.5 * PI, -u, -v], vec!['L', 'R', 'S', 'R']);
    }

    let (flag, t, u, v) = lrsr(x, -y, -phi);
    if flag {
        set_path(paths, vec![t, -0.5 * PI, u, v], vec!['R', 'L', 'S', 'L']);
    }

    let (flag, t, u, v) = lrsr(-x, -y, phi);
    if flag {
        set_path(paths, vec![-t, 0.5 * PI, -u, -v], vec!['R', 'L', 'S', 'L']);
    }

    // backwards
    let xb = x * phi.cos() + y * phi.sin();
    let yb = x * phi.sin() - y * phi.cos();

    let (flag, t, u, v) = lrsl(xb, yb, phi);
    if flag {
        set_path(paths, vec![v, u, -0.5 * PI, t], vec!['L', 'S', 'R', 'L']);
    }

    let (flag, t, u, v) = lrsl(-xb, yb, -phi);
    if flag {
        set_path(paths, vec![-v, -u, 0.5 * PI, -t], vec!['L', 'S', 'R', 'L']);
    }

    let (flag, t, u, v) = lrsl(xb, -yb, -phi);
    if flag {
        set_path(paths, vec![v, u, -0.5 * PI, t], vec!['R', 'S', 'L', 'R']);
    }

    let (flag, t, u, v) = lrsl(-xb, -yb, phi);
    if flag {
        set_path(paths, vec![-v, -u, 0.5 * PI, -t], vec!['R', 'S', 'L', 'R']);
    }

    let (flag, t, u, v) = lrsr(xb, yb, phi);
    if flag {
        set_path(paths, vec![v, u, -0.5 * PI, t], vec!['R', 'S', 'R', 'L']);
    }

    let (flag, t, u, v) = lrsr(-xb, yb, -phi);
    if flag {
        set_path(paths, vec![-v, -u, 0.5 * PI, -t], vec!['R', 'S', 'R', 'L']);
    }

    let (flag, t, u, v) = lrsr(xb, -yb, -phi);
    if flag {
        set_path(paths, vec![v, u, -0.5 * PI, t], vec!['L', 'S', 'L', 'R']);
    }

    let (flag, t, u, v) = lrsr(-xb, -yb, phi);
    if flag {
        set_path(paths, vec![-v, -u, 0.5 * PI, -t], vec!['L', 'S', 'L', 'R']);
    }
}

fn lrslr(x: f64, y: f64, phi: f64) -> (bool, f64, f64, f64) {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let (rho, _theta) = r(xi, eta);

    if rho >= 2.0 {
        let u = 4.0 - (rho.powi(2) - 4.0).sqrt();
        if u <= 0. {
            let t = m(((4.0 - u) * xi - 2.0 * eta).atan2(-2.0 * xi + (u - 4.0) * eta));
            let v = m(t - phi);
            if (t >= 0.0) & (v >= 0.0) {
                return (true, t, u, v);
            }
        }
    }
    (false, 0., 0., 0.)
}

fn ccscc(x: f64, y: f64, phi: f64, paths: &mut Vec<RSPath>) {
    let (flag, t, u, v) = lrslr(x, y, phi);
    if flag {
        set_path(
            paths,
            vec![t, -0.5 * PI, u, -0.5 * PI, v],
            vec!['L', 'R', 'S', 'L', 'R'],
        );
    }

    let (flag, t, u, v) = lrslr(-x, y, -phi);
    if flag {
        set_path(
            paths,
            vec![-t, 0.5 * PI, -u, 0.5 * PI, -v],
            vec!['L', 'R', 'S', 'L', 'R'],
        );
    }

    let (flag, t, u, v) = lrslr(x, -y, -phi);
    if flag {
        set_path(
            paths,
            vec![t, -0.5 * PI, u, -0.5 * PI, v],
            vec!['R', 'L', 'S', 'R', 'L'],
        );
    }

    let (flag, t, u, v) = lrslr(-x, -y, phi);
    if flag {
        set_path(
            paths,
            vec![-t, 0.5 * PI, -u, 0.5 * PI, -v],
            vec!['R', 'L', 'S', 'R', 'L'],
        );
    }
}
fn generate_local_course(
    total_length: f64,
    lengths: &Vec<f64>,
    mode: &Vec<char>,
    maxc: f64,
    step_size: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<isize>) {
    let point_num = (total_length / step_size) as usize + lengths.len() + 3;
    let mut px = vec![0.0; point_num];
    let mut py = vec![0.0; point_num];
    let mut pyaw = vec![0.0; point_num];
    let mut directions: Vec<isize> = vec![0; point_num];
    let mut ind: usize = 1;

    if lengths[0] > 0.0 {
        directions[0] = 1;
    } else {
        directions[0] = -1;
    }
    let mut ll = 0.0;
    mode.iter()
        .zip(lengths.iter())
        .enumerate()
        .for_each(|(i, (&m, &l))| {
            let d;
            if l > 0.0 {
                d = step_size;
            } else {
                d = -step_size;
            }
            let ox = px[ind];
            let oy = py[ind];
            let oyaw = pyaw[ind];
            ind -= 1;
            let mut pd;
            if i >= 1 {
                if lengths[i - 1] * lengths[i] > 0. {
                    pd = -d - ll;
                } else {
                    pd = d - ll;
                }
            } else {
                pd = d - ll;
            }
            while pd.abs() <= l.abs() {
                ind += 1;
                interpolate(
                    pd,
                    m,
                    maxc,
                    ox,
                    oy,
                    oyaw,
                    &mut px[ind],
                    &mut py[ind],
                    &mut pyaw[ind],
                    &mut directions[ind],
                );
                pd += d;
            }
            ll = l - pd - d;
            ind += 1;
            interpolate(
                l,
                m,
                maxc,
                ox,
                oy,
                oyaw,
                &mut px[ind],
                &mut py[ind],
                &mut pyaw[ind],
                &mut directions[ind],
            );
        });
    while *px.last().unwrap() == 0.0 {
        px.pop();
        py.pop();
        pyaw.pop();
        directions.pop();
    }
    (px, py, pyaw, directions)
}

fn interpolate(
    l: f64,
    m: char,
    maxc: f64,
    ox: f64,
    oy: f64,
    oyaw: f64,
    px: &mut f64,
    py: &mut f64,
    pyaw: &mut f64,
    directions: &mut isize,
) {
    if m == 'S' {
        *px = ox + l / maxc * oyaw.cos();
        *py = oy + l / maxc * oyaw.sin();
        *pyaw = oyaw;
    } else {
        let ldx = l.sin() / maxc;
        let ldy;
        if m == 'L' {
            ldy = (1.0 - l.cos()) / maxc;
        } else {
            // m == 'R'
            ldy = -(1. - l.cos()) / maxc;
        }
        let gdx = (-oyaw).cos() * ldx + (-oyaw).sin() * ldy;
        let gdy = -(-oyaw).sin() * ldx + (-oyaw).cos() * ldy;
        *px = ox + gdx;
        *py = oy + gdy;
    }
    if m == 'L' {
        *pyaw = oyaw + l;
    } else if m == 'R' {
        *pyaw = oyaw - l;
    }
    if l > 0.0 {
        *directions = 1;
    } else {
        *directions = -1;
    }
}

fn generate_path(q0: &[f64; 3], q1: &[f64; 3], maxc: f64) -> Vec<RSPath> {
    let dx = q1[0] - q0[0];
    let dy = q1[1] - q0[1];
    let dth = q1[2] - q0[2];
    let c = q0[2].cos();
    let s = q0[2].sin();
    let x = (c * dx + s * dy) * maxc;
    let y = (-s * dx + c * dy) * maxc;

    let mut paths = Vec::<RSPath>::new();
    scs(x, y, dth, &mut paths);
    csc(x, y, dth, &mut paths);
    ccc(x, y, dth, &mut paths);
    cccc(x, y, dth, &mut paths);
    ccsc(x, y, dth, &mut paths);
    ccscc(x, y, dth, &mut paths);

    paths
}

pub fn pi_2_pi(theta: f64) -> f64 {
    let mut theta = theta;
    while theta > PI {
        theta -= 2.0 * PI;
    }

    while theta < -PI {
        theta += 2.0 * PI;
    }

    theta
}

fn r(x: f64, y: f64) -> (f64, f64) {
    let r = x.hypot(y);
    let theta = y.atan2(x);

    (r, theta)
}

fn m(theta: f64) -> f64 {
    let mut phi = theta % (2.0 * PI);
    if phi < -PI {
        phi += 2.0 * PI;
    }
    if phi > PI {
        phi -= 2.0 * PI;
    }

    phi
}
