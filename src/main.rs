mod astar;
mod data_struct;
mod hybrid_astar;
mod reeds_shepp;
mod util;
use std::time::Instant;

use astar::astar_planning;
use hybrid_astar::{Config, hybrid_astar_planning};
use reeds_shepp::{Car, calc_optimal_path, draw_car};
use rerun::external::glam::Vec2;


fn setup_astar_env() -> (Vec<f64>, Vec<f64>) {
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

fn design_hybrid_astar_obstacles(x: usize, y: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ox = Vec::new();
    let mut oy = Vec::new();

    for i in 0..x {
        ox.push(i as f64);
        oy.push(0.);
    }
    for i in 0..x {
        ox.push(i as f64);
        oy.push((y - 1) as f64);
    }
    for i in 0..y {
        ox.push(0.);
        oy.push(i as f64);
    }
    for i in 0..y {
        ox.push((x - 1) as f64);
        oy.push(i as f64);
    }
    for i in 10..21 {
        ox.push(i as f64);
        oy.push(15.);
    }
    for i in 0..15 {
        ox.push(20.);
        oy.push(i as f64);
    }
    for i in 15..30 {
        ox.push(30.);
        oy.push(i as f64);
    }
    for i in 0..16 {
        ox.push(40.);
        oy.push(i as f64);
    }

    (ox, oy)
}

fn main() {
    {
        let rec = rerun::RecordingStreamBuilder::new("reeds-shepp")
            .spawn()
            .unwrap();

        println!("reeds shepp start");
        let states = [
            [0., 0., 0.],
            [10., 10., -90.],
            [20., 5., 60.],
            [30., 10., 120.],
            [35., -5., 30.],
            [25., -10., -120.],
            [15., -15., 100.],
            [0., -10., -90.],
        ];
        let points: Vec<Vec2> = states
            .into_iter()
            .map(|p| Vec2::new(p[0] as f32, p[1] as f32))
            .collect();
        let vectors: Vec<Vec2> = states
            .into_iter()
            .map(|p| {
                Vec2::new(
                    f64::to_radians(p[2]).cos() as f32,
                    f64::to_radians(p[2]).sin() as f32,
                )
            })
            .collect();
        let _ = rec.log(
            "checkpoints",
            &rerun::Arrows2D::from_vectors(vectors).with_origins(points),
        );

        let maxc = 0.1;
        let mut path_x: Vec<f64> = Vec::new();
        let mut path_y: Vec<f64> = Vec::new();
        let mut yaw: Vec<f64> = Vec::new();
        let mut directions: Vec<isize> = Vec::new();
        let car = reeds_shepp::Car::new(4.5, 1.0, 3.0, 3.5);
        for i in 0..states.len() - 1 {
            let rspath = calc_optimal_path(
                states[i][0],
                states[i][1],
                states[i][2].to_radians(),
                states[i + 1][0],
                states[i + 1][1],
                states[i + 1][2].to_radians(),
                maxc,
                0.2,
            );
            path_x.extend(rspath.traj_x);
            path_y.extend(rspath.traj_y);
            yaw.extend(rspath.traj_yaw);
            directions.extend(rspath.traj_dirs);

            let line: Vec<Vec2> = path_x
                .iter()
                .zip(path_y.iter())
                .map(|(&a, &b)| Vec2::new(a as f32, b as f32))
                .collect();
            let _ = rec.log("path", &rerun::LineStrips2D::new([line]));
        }
        for k in 0..path_x.len() {
            reeds_shepp::draw_car(path_x[k], path_y[k], yaw[k], &car, &rec);
        }
    }
    {
        let rec = rerun::RecordingStreamBuilder::new("astar").spawn().unwrap();
        println!("astar start");
        let sx = 10.0; // [m]
        let sy = 10.0; // [m]
        let gx = 50.0; // [m]
        let gy = 50.0; // [m]
        let _ = rec.log(
            "start",
            &rerun::Points2D::new([Vec2::new(sx as f32, sy as f32)]),
        );
        let _ = rec.log(
            "goal",
            &rerun::Points2D::new([Vec2::new(gx as f32, gy as f32)]),
        );

        let robot_radius = 2.0;
        let grid_resolution = 1.0;
        let (ox, oy) = setup_astar_env();
        let obs: Vec<Vec2> = ox
            .iter()
            .zip(oy.iter())
            .map(|(&a, &b)| Vec2::new(a as f32, b as f32))
            .collect();
        let _ = rec.log("obstacles", &rerun::Points2D::new(obs));

        let t0 = Instant::now();
        let path = astar_planning(
            sx,
            sy,
            gx,
            gy,
            &ox,
            &oy,
            grid_resolution,
            robot_radius,
            Some(&rec),
        );
        let t1 = Instant::now();
        println!("running T: {:?}", t1.duration_since(t0));
        match path {
            Some((pathx, pathy)) => {
                let line: Vec<Vec2> = pathx
                    .iter()
                    .zip(pathy.iter())
                    .map(|(&a, &b)| Vec2::new(a as f32, b as f32))
                    .collect();
                let _ = rec.log("path", &rerun::LineStrips2D::new([line]));
            }
            None => {
                println!("Searching failed!");
            }
        }
    }

    {
        let rec = rerun::RecordingStreamBuilder::new("hybrid_astar")
            .spawn()
            .unwrap();

        println!("hybrid astar start!");
        let the_car = Car::new(4.5, 1.0, 3.0, 3.5);
        let this_conf = Config::new(
            2.0,
            15.0_f64.to_radians(),
            0.4,
            20,
            5,
            1.4,
            100.0,
            5.0,
            5.0,
            1.0,
            15.0,
            the_car,
            0.6,
        );

        let (x, y) = (51, 31);
        let (sx, sy, syaw0) = (10.0, 7.0, 120.0_f64.to_radians());
        let (gx, gy, gyaw0) = (45.0, 20.0, 90.0_f64.to_radians());

        let _ = rec.log(
            "start",
            &rerun::Points2D::new([Vec2::new(sx as f32, sy as f32)]),
        );
        let _ = rec.log(
            "goal",
            &rerun::Points2D::new([Vec2::new(gx as f32, gy as f32)]),
        );
        let (ox, oy) = design_hybrid_astar_obstacles(x, y);

        let obs: Vec<Vec2> = ox
            .iter()
            .zip(oy.iter())
            .map(|(&a, &b)| Vec2::new(a as f32, b as f32))
            .collect();
        let sizes = vec![[1.0_f32, 1.0_f32]; obs.len()];
        let _ = rec.log(
            "obstacles",
            &rerun::Boxes2D::from_centers_and_sizes(obs, sizes),
        );

        let t0 = Instant::now();
        let path = hybrid_astar_planning(
            sx,
            sy,
            syaw0,
            gx,
            gy,
            gyaw0,
            &ox,
            &oy,
            &this_conf,
            Some(&rec),
        );

        let t1 = Instant::now();
        println!("running T: {:?}", t1.duration_since(t0));

        match path {
            Some(p) => {
                println!("Done!");
                let x = p.x;
                let y = p.y;
                let yaw = p.yaw;
                println!("Total cost: {:?}", p.cost);
                let line: Vec<Vec2> = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| Vec2::new(a as f32, b as f32))
                    .collect();
                let _ = rec.log("path", &rerun::LineStrips2D::new([line]));
                for k in 0..x.len() {
                    draw_car(x[k], y[k], yaw[k], &this_conf.car, &rec);
                }
            }
            None => {
                println!("Searching failed!");
            }
        }
    }
}
