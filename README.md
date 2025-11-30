# TEB Local Planner (Standalone C++ with Pybind11)

The teb_local_planner implements a Timed Elastic Band approach for trajectory optimization.
The underlying method locally optimizes the robot's trajectory with respect to trajectory execution time,
separation from obstacles and compliance with kinodynamic constraints at runtime.

**This is a standalone C++ version without ROS dependencies, with Python bindings via pybind11.**

## Features

- Time-optimal trajectory planning
- Obstacle avoidance (Point, Circular, Line, Polygon)
- Dynamic obstacle support
- Differential drive / Car-like robot support
- Homotopy Class Planning (multiple trajectory exploration)
- Python bindings (pybind11)

## Dependencies

- Eigen3
- Boost
- g2o (graph optimization)
- SuiteSparse (CSparse)
- pybind11 (optional, for Python bindings)

### Ubuntu Installation

```bash
sudo apt install libeigen3-dev libboost-dev libsuitesparse-dev pybind11-dev

# g2o (build from source)
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

## Build

```bash
git clone <repo_url>
cd teb_local_planner
mkdir build && cd build
cmake ../teb_local_planner
make -j$(nproc)
```

Output:
- `libteb_local_planner.so` - C++ library
- `pyteb.cpython-*.so` - Python module

## Python Usage

```python
import sys
sys.path.insert(0, '/path/to/build')
import pyteb

# Configuration
cfg = pyteb.TebConfig()
cfg.robot.max_vel_x = 0.5
cfg.robot.max_vel_theta = 1.0
cfg.robot_model = pyteb.CircularRobotFootprint(0.2)

# Obstacles
obstacles = pyteb.ObstacleContainer()
obstacles.add(pyteb.CircularObstacle(2.0, 0.0, 0.5))

# Plan
planner = pyteb.TebOptimalPlanner(cfg, obstacles)
start = pyteb.PoseSE2(0, 0, 0)
goal = pyteb.PoseSE2(5, 0, 0)
planner.plan(start, goal, pyteb.Velocity2D())

# Result
teb = planner.teb()
print(f"Poses: {teb.size_poses()}")
print(f"Time: {teb.get_sum_of_all_time_diffs():.2f}s")
```

## Visualization Test

```bash
cd teb_local_planner/python
python3 test_visualization.py --mode static               # Static visualization
python3 test_visualization.py --mode homotopy             # Homotopy classes
python3 test_visualization.py --mode homotopy_interactive # Interactive demo
```

## Main Classes

| Class | Description |
|-------|-------------|
| `TebConfig` | Planner configuration |
| `TebOptimalPlanner` | Single trajectory optimization |
| `HomotopyClassPlanner` | Multiple trajectory exploration |
| `PoseSE2` | 2D pose (x, y, theta) |
| `Velocity2D` | Velocity (vx, vy, omega) |
| `ObstacleContainer` | Obstacle container |


## Acknowledgements

This project is based on [teb_local_planner](https://github.com/rst-tu-dortmund/teb_local_planner).
