pub fn rot2d(p: &[f64; 2], th: f64) -> (f64, f64) {
    let x = p[0] * th.cos() - p[1] * th.sin();
    let y = p[0] * th.sin() + p[1] * th.cos();
    (x, y)
}
