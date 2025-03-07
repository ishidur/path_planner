# Path Planner

Demo code for path planning algorithms written in Rust/Python (for study purposes).  
[Rerun](https://rerun.io/) is used for the visualization.

## Algorithms

- A\*  
  ![astar](https://github.com/ishidur/path_planner/blob/main/figure/astar.gif?raw=true)
- Hybrid A\*  
  ![hybrid astar](https://github.com/ishidur/path_planner/blob/main/figure/hybrid_astar.gif?raw=true)
- (RRT)  
- (Theta\*)  

## Rust

To run A\* and Hybrid A\*  (Rerun Viewer will spawn automatically)
```bash
cargo run -r
```

## Python

```bash
uv sync
```

To run A\* (Rerun Viewer will spawn automatically)

```bash
uv run astar.py
```
To run Hybrid A\* (Rerun Viewer will spawn automatically)

```bash
uv run hybrid_astar.py
```

## References
https://github.com/zhm-real/MotionPlanning  
https://github.com/zhm-real/CurvesGenerator
