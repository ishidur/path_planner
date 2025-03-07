mod astar;
mod data_struct;
mod hybrid_astar;
mod reeds_shepp;
mod util;
use astar::{astar_planning, get_env};
use hybrid_astar::{Config, design_obstacles, draw_car, hybrid_astar_planning};
use reeds_shepp::pi_2_pi;
use rerun::external::glam::Vec2;
use std::time::Instant;

fn main() {
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
        let (ox, oy) = get_env();
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
            &rec,
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
        rec.set_time_seconds("step", 0.);

        println!("hybrid astar start!");
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
            4.5,
            1.0,
            3.0,
            0.7 * 3.0,
            3.5,
            0.5,
            1.0,
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
        let (ox, oy) = design_obstacles(x, y);

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
        let path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0, &ox, &oy, &this_conf, &rec);

        let t1 = Instant::now();
        println!("running T: {:?}", t1.duration_since(t0));

        match path {
            Some(p) => {
                println!("Done!");
                let x = p.x;
                let y = p.y;
                let yaw = p.yaw;
                let direction = p.direction;
                println!("Total cost: {:?}", p.cost);
                let line: Vec<Vec2> = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| Vec2::new(a as f32, b as f32))
                    .collect();
                let _ = rec.log("path", &rerun::LineStrips2D::new([line]));
                for k in 0..x.len() {
                    rec.set_time_seconds("step", k as f64);
                    let steer;
                    if k < x.len() - 2 {
                        let dy = (yaw[k + 1] - yaw[k]) / this_conf.move_step;
                        steer = pi_2_pi(-this_conf.wb * dy / direction[k] as f64);
                    } else {
                        steer = 0.0;
                    }
                    draw_car(x[k], y[k], yaw[k], steer, &this_conf, &rec);
                }
            }
            None => {
                println!("Searching failed!");
            }
        }
    }
}
