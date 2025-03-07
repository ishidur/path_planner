#[derive(Debug, Clone)]
pub struct Node {
    pub xind: usize,
    pub yind: usize,
    pub yawind: usize,
    pub direction: isize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub yaw: Vec<f64>,
    pub directions: Vec<isize>,
    pub steer: f64,
    pub cost: f64,
    pub pind: isize,
}

impl Node {
    pub fn new(
        xind: usize,
        yind: usize,
        yawind: usize,
        direction: isize,
        x: Vec<f64>,
        y: Vec<f64>,
        yaw: Vec<f64>,
        directions: Vec<isize>,
        steer: f64,
        cost: f64,
        pind: isize,
    ) -> Self {
        Node {
            xind,
            yind,
            yawind,
            direction,
            x,
            y,
            yaw,
            directions,
            steer,
            cost,
            pind,
        }
    }
}
