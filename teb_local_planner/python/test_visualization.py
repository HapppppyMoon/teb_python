#!/usr/bin/env python3
"""
TEB Local Planner - Visualization Test
Simple example demonstrating trajectory planning and visualization
"""

import sys
sys.path.insert(0, '/media/isr12/ext_hdd/git_repos/teb_local_planner/build_standalone')

import pyteb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrow
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation


def create_config():
    """Create and configure TEB planner settings"""
    cfg = pyteb.TebConfig()

    # Robot parameters
    cfg.robot.max_vel_x = 0.5
    cfg.robot.max_vel_x_backwards = 0.2
    cfg.robot.max_vel_theta = 1.0
    cfg.robot.acc_lim_x = 0.5
    cfg.robot.acc_lim_theta = 0.5

    # Trajectory parameters
    cfg.trajectory.dt_ref = 0.3
    cfg.trajectory.min_samples = 3
    cfg.trajectory.max_samples = 100

    # Obstacle parameters
    cfg.obstacles.min_obstacle_dist = 0.25
    cfg.obstacles.inflation_dist = 0.3

    # Optimization parameters
    cfg.optim.no_inner_iterations = 5
    cfg.optim.no_outer_iterations = 4
    cfg.optim.weight_obstacle = 100
    cfg.optim.weight_optimaltime = 1

    # Robot footprint (circular)
    cfg.robot_model = pyteb.CircularRobotFootprint(0.2)

    return cfg


def create_obstacles():
    """Create obstacle container with various obstacle types"""
    obstacles = pyteb.ObstacleContainer()

    # Point obstacles
    obstacles.add(pyteb.PointObstacle(1.5, 0.8))
    obstacles.add(pyteb.PointObstacle(2.5, -0.5))

    # Circular obstacles
    obstacles.add(pyteb.CircularObstacle(1.0, -0.3, 0.3))
    obstacles.add(pyteb.CircularObstacle(2.0, 0.5, 0.25))
    obstacles.add(pyteb.CircularObstacle(3.5, 0.0, 0.4))

    # Line obstacle
    obstacles.add(pyteb.LineObstacle(2.8, -1.0, 3.2, -0.3))

    return obstacles


def plot_obstacles(ax, obstacles):
    """Plot all obstacles"""
    patches = []

    for i in range(len(obstacles)):
        obs = obstacles[i]
        centroid = obs.get_centroid()

        # Check obstacle type by trying different methods
        if isinstance(obs, pyteb.CircularObstacle):
            circle = Circle((centroid[0], centroid[1]), obs.radius,
                          facecolor='red', edgecolor='darkred', alpha=0.6)
            patches.append(circle)
        elif isinstance(obs, pyteb.PointObstacle):
            circle = Circle((centroid[0], centroid[1]), 0.1,
                          facecolor='orange', edgecolor='darkorange', alpha=0.8)
            patches.append(circle)
        elif isinstance(obs, pyteb.LineObstacle):
            start = obs.start()
            end = obs.end()
            ax.plot([start[0], end[0]], [start[1], end[1]],
                   'r-', linewidth=3, solid_capstyle='round')
        else:
            # Generic obstacle - plot as point
            ax.plot(centroid[0], centroid[1], 'rx', markersize=10)

    if patches:
        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)


def plot_robot(ax, pose, radius=0.2, color='blue'):
    """Plot robot as a circle with direction arrow"""
    x, y, theta = pose.x, pose.y, pose.theta

    # Robot body
    robot = Circle((x, y), radius, facecolor=color, edgecolor='darkblue', alpha=0.7)
    ax.add_patch(robot)

    # Direction arrow
    arrow_len = radius * 1.5
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    ax.arrow(x, y, dx, dy, head_width=0.08, head_length=0.05,
             fc='darkblue', ec='darkblue')


def plot_trajectory(ax, teb, color='green', alpha=0.8):
    """Plot trajectory from TimedElasticBand"""
    poses = teb.get_all_poses()

    if len(poses) < 2:
        return

    xs = [p.x for p in poses]
    ys = [p.y for p in poses]

    # Plot trajectory line
    ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=alpha, label='Trajectory')

    # Plot waypoints
    ax.scatter(xs, ys, c=color, s=30, alpha=alpha, zorder=5)

    # Plot orientation arrows at intervals
    for i in range(0, len(poses), max(1, len(poses)//10)):
        p = poses[i]
        dx = 0.15 * np.cos(p.theta)
        dy = 0.15 * np.sin(p.theta)
        ax.arrow(p.x, p.y, dx, dy, head_width=0.05, head_length=0.03,
                fc=color, ec=color, alpha=alpha*0.7)


def plot_velocity_profile(ax, planner):
    """Plot velocity profile"""
    teb = planner.teb()
    full_traj = planner.get_full_trajectory()

    times = [tp.time_from_start for tp in full_traj]
    vx = [tp.velocity.vx for tp in full_traj]
    omega = [tp.velocity.omega for tp in full_traj]

    ax.plot(times, vx, 'b-', label='vx [m/s]', linewidth=2)
    ax.plot(times, omega, 'r-', label='omega [rad/s]', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Velocity Profile')


def homotopy_visualization():
    """Visualization with HomotopyClassPlanner - explores multiple trajectory classes"""
    print("=" * 50)
    print("TEB Local Planner - Homotopy Class Visualization")
    print("=" * 50)

    # Setup
    cfg = create_config()

    # Enable homotopy class planning
    cfg.hcp.enable_homotopy_class_planning = True
    cfg.hcp.max_number_classes = 5
    cfg.hcp.selection_cost_hysteresis = 1.0
    cfg.hcp.roadmap_graph_no_samples = 15
    cfg.hcp.roadmap_graph_area_width = 5.0

    # Create obstacle in the MIDDLE - NOT on the path (y=0)
    # This creates two homotopy classes: go above or go below
    obstacles = pyteb.ObstacleContainer()
    obstacles.add(pyteb.CircularObstacle(2.5, 0.0, 0.5))   # Center obstacle

    # Create HomotopyClassPlanner
    planner = pyteb.HomotopyClassPlanner(cfg, obstacles)

    # Define start and goal
    start = pyteb.PoseSE2(0.0, 0.0, 0.0)
    goal = pyteb.PoseSE2(5.0, 0.0, 0.0)
    start_vel = pyteb.Velocity2D(0.0, 0.0, 0.0)

    # Create initial plan that goes ABOVE the obstacle
    # 첫 점 = start, 마지막 점 = goal, 중간은 arc
    initial_plan = [start]
    num_points = 15
    for i in range(1, num_points):
        t = i / num_points
        x = start.x + t * (goal.x - start.x)
        y = 1.2 * np.sin(np.pi * t)
        # 중간 점 theta = 접선 방향
        dy_dt = 1.2 * np.pi * np.cos(np.pi * t)
        dx_dt = goal.x - start.x
        theta = np.arctan2(dy_dt, dx_dt)
        initial_plan.append(pyteb.PoseSE2(x, y, theta))
    initial_plan.append(goal)

    # Plan using initial_plan
    print(f"\nPlanning from ({start.x}, {start.y}) to ({goal.x}, {goal.y})...")
    print(f"Initial plan goes ABOVE the obstacle")
    success = planner.plan(initial_plan, start_vel)

    print(f"Planning success: {success}")
    print(f"Initialized: {planner.is_initialized()}")

    # Get all trajectory candidates
    trajectories = planner.get_trajectory_container()
    print(f"Number of homotopy classes found: {len(trajectories)}")

    best_teb = planner.best_teb()
    best_idx = planner.best_teb_idx()
    print(f"Best trajectory index: {best_idx}")

    if best_teb:
        teb = best_teb.teb()
        print(f"Best trajectory poses: {teb.size_poses()}")
        print(f"Best trajectory time: {teb.get_sum_of_all_time_diffs():.3f} s")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: All trajectories
    ax1 = axes[0]
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-2, 2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'Homotopy Class Planning ({len(trajectories)} classes)')

    # Plot obstacles
    plot_obstacles(ax1, obstacles)

    # Plot all trajectory candidates with different colors
    colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
    for i, teb_planner in enumerate(trajectories):
        teb = teb_planner.teb()
        color = colors[i % len(colors)]
        alpha = 1.0 if i == best_idx else 0.4
        linewidth = 3 if i == best_idx else 1.5

        poses = teb.get_all_poses()
        xs = [p.x for p in poses]
        ys = [p.y for p in poses]

        label = f'Class {i+1}' + (' (best)' if i == best_idx else '')
        ax1.plot(xs, ys, '-', color=color, linewidth=linewidth, alpha=alpha, label=label)
        ax1.scatter(xs[::3], ys[::3], c=color, s=15, alpha=alpha)

    # Plot start and goal
    plot_robot(ax1, start, radius=0.15, color='green')
    ax1.annotate('Start', (start.x, start.y - 0.35), ha='center', fontsize=10)
    plot_robot(ax1, goal, radius=0.15, color='red')
    ax1.annotate('Goal', (goal.x, goal.y - 0.35), ha='center', fontsize=10)

    ax1.legend(loc='upper right', fontsize=9)

    # Right plot: Best trajectory velocity profile
    ax2 = axes[1]
    if best_teb:
        full_traj = best_teb.get_full_trajectory()
        times = [tp.time_from_start for tp in full_traj]
        vx = [tp.velocity.vx for tp in full_traj]
        omega = [tp.velocity.omega for tp in full_traj]

        ax2.plot(times, vx, 'b-', label='vx [m/s]', linewidth=2)
        ax2.plot(times, omega, 'r-', label='omega [rad/s]', linewidth=2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Best Trajectory - Velocity Profile')

    plt.tight_layout()
    plt.savefig('/media/isr12/ext_hdd/git_repos/teb_local_planner/build_standalone/teb_homotopy.png', dpi=150)
    print(f"\nSaved to: teb_homotopy.png")
    plt.show()


def static_visualization():
    """Static visualization of planned trajectory"""
    print("=" * 50)
    print("TEB Local Planner - Static Visualization")
    print("=" * 50)

    # Setup
    cfg = create_config()
    obstacles = create_obstacles()

    # Create planner
    planner = pyteb.TebOptimalPlanner(cfg, obstacles)

    # Define start and goal
    start = pyteb.PoseSE2(0.0, 0.0, 0.0)
    goal = pyteb.PoseSE2(4.5, 0.0, 0.0)
    start_vel = pyteb.Velocity2D(0.0, 0.0, 0.0)

    # Plan
    print(f"\nPlanning from ({start.x}, {start.y}) to ({goal.x}, {goal.y})...")
    success = planner.plan(start, goal, start_vel)

    print(f"Planning success: {success}")
    print(f"Optimized: {planner.is_optimized()}")
    print(f"Cost: {planner.get_current_cost():.4f}")

    teb = planner.teb()
    print(f"Trajectory poses: {teb.size_poses()}")
    print(f"Total time: {teb.get_sum_of_all_time_diffs():.3f} s")
    print(f"Total distance: {teb.get_accumulated_distance():.3f} m")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Trajectory
    ax1 = axes[0]
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(-2, 2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('TEB Trajectory Planning')

    # Plot obstacles
    plot_obstacles(ax1, obstacles)

    # Plot trajectory
    plot_trajectory(ax1, teb)

    # Plot start and goal
    plot_robot(ax1, start, radius=0.2, color='green')
    ax1.annotate('Start', (start.x, start.y - 0.4), ha='center', fontsize=10)

    plot_robot(ax1, goal, radius=0.2, color='red')
    ax1.annotate('Goal', (goal.x, goal.y - 0.4), ha='center', fontsize=10)

    # Right plot: Velocity profile
    ax2 = axes[1]
    plot_velocity_profile(ax2, planner)

    plt.tight_layout()
    plt.savefig('/media/isr12/ext_hdd/git_repos/teb_local_planner/build_standalone/teb_result.png', dpi=150)
    print(f"\nSaved to: teb_result.png")
    plt.show()


def animated_visualization():
    """Animated visualization of robot following trajectory"""
    print("=" * 50)
    print("TEB Local Planner - Animated Visualization")
    print("=" * 50)

    # Setup
    cfg = create_config()
    obstacles = create_obstacles()

    # Create planner
    planner = pyteb.TebOptimalPlanner(cfg, obstacles)

    # Define start and goal
    start = pyteb.PoseSE2(0.0, 0.0, 0.0)
    goal = pyteb.PoseSE2(4.5, 0.0, 0.0)
    start_vel = pyteb.Velocity2D(0.0, 0.0, 0.0)

    # Plan
    success = planner.plan(start, goal, start_vel)
    if not success:
        print("Planning failed!")
        return

    teb = planner.teb()
    poses = teb.get_all_poses()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('TEB Trajectory - Animation')

    # Plot static elements
    plot_obstacles(ax, obstacles)

    # Plot full trajectory (faded)
    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    ax.plot(xs, ys, '--', color='lightgreen', linewidth=1, alpha=0.5)

    # Goal marker
    goal_circle = Circle((goal.x, goal.y), 0.2, facecolor='red',
                         edgecolor='darkred', alpha=0.5)
    ax.add_patch(goal_circle)
    ax.annotate('Goal', (goal.x, goal.y - 0.4), ha='center', fontsize=10)

    # Robot (will be updated)
    robot_circle = Circle((start.x, start.y), 0.2, facecolor='blue',
                          edgecolor='darkblue', alpha=0.7)
    ax.add_patch(robot_circle)

    # Direction arrow
    arrow = ax.arrow(start.x, start.y, 0.3, 0, head_width=0.1, head_length=0.05,
                    fc='darkblue', ec='darkblue')

    # Trajectory trace
    trace_line, = ax.plot([], [], 'g-', linewidth=2, alpha=0.8)

    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.5))

    def init():
        trace_line.set_data([], [])
        time_text.set_text('')
        return robot_circle, trace_line, time_text

    def animate(frame):
        if frame >= len(poses):
            return robot_circle, trace_line, time_text

        pose = poses[frame]

        # Update robot position
        robot_circle.center = (pose.x, pose.y)

        # Update direction arrow (remove old, add new)
        global arrow
        arrow.remove()
        dx = 0.3 * np.cos(pose.theta)
        dy = 0.3 * np.sin(pose.theta)
        arrow = ax.arrow(pose.x, pose.y, dx, dy, head_width=0.1, head_length=0.05,
                        fc='darkblue', ec='darkblue')

        # Update trace
        trace_x = [p.x for p in poses[:frame+1]]
        trace_y = [p.y for p in poses[:frame+1]]
        trace_line.set_data(trace_x, trace_y)

        # Update time
        time_diffs = teb.get_all_time_diffs()
        if frame < len(time_diffs):
            elapsed = sum(time_diffs[:frame])
        else:
            elapsed = teb.get_sum_of_all_time_diffs()
        time_text.set_text(f'Time: {elapsed:.2f}s\nPose: {frame+1}/{len(poses)}')

        return robot_circle, trace_line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(poses), interval=100, blit=False)

    # Save animation
    print("Saving animation (this may take a moment)...")
    anim.save('/media/isr12/ext_hdd/git_repos/teb_local_planner/build_standalone/teb_animation.gif',
              writer='pillow', fps=10)
    print("Saved to: teb_animation.gif")

    plt.show()


def interactive_replanning():
    """Interactive demo with dynamic obstacle and replanning"""
    print("=" * 50)
    print("TEB Local Planner - Interactive Replanning Demo")
    print("=" * 50)
    print("Click to add obstacles, press 'r' to replan, 'c' to clear, 'q' to quit")

    # Setup
    cfg = create_config()
    obstacles = pyteb.ObstacleContainer()

    # Add initial obstacles
    obstacles.add(pyteb.CircularObstacle(2.0, 0.3, 0.3))
    obstacles.add(pyteb.CircularObstacle(3.0, -0.2, 0.25))

    # Create planner
    planner = pyteb.TebOptimalPlanner(cfg, obstacles)

    # Define start and goal
    start = pyteb.PoseSE2(0.0, 0.0, 0.0)
    goal = pyteb.PoseSE2(5.0, 0.0, 0.0)

    # Initial plan
    planner.plan(start, goal, pyteb.Velocity2D())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    def update_plot():
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('TEB Interactive Demo (click to add obstacles, r=replan, c=clear)')

        # Plot obstacles
        plot_obstacles(ax, obstacles)

        # Plot trajectory
        teb = planner.teb()
        if teb.is_init():
            plot_trajectory(ax, teb)

        # Plot start and goal
        plot_robot(ax, start, radius=0.2, color='green')
        plot_robot(ax, goal, radius=0.2, color='red')

        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click
            obstacles.add(pyteb.CircularObstacle(event.xdata, event.ydata, 0.2))
            print(f"Added obstacle at ({event.xdata:.2f}, {event.ydata:.2f})")
            planner.plan(start, goal, pyteb.Velocity2D())
            update_plot()

    def on_key(event):
        if event.key == 'r':
            print("Replanning...")
            planner.clear_planner()
            planner.plan(start, goal, pyteb.Velocity2D())
            update_plot()
        elif event.key == 'c':
            print("Clearing obstacles...")
            obstacles.clear()
            planner.clear_planner()
            planner.plan(start, goal, pyteb.Velocity2D())
            update_plot()
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()
    plt.show()


def interactive_homotopy():
    """Interactive demo with HomotopyClassPlanner"""
    print("=" * 50)
    print("TEB Local Planner - Interactive Homotopy Demo")
    print("=" * 50)
    print("Click to add obstacles, press 'r' to replan, 'c' to clear, 'q' to quit")

    # Setup
    cfg = create_config()
    cfg.hcp.enable_homotopy_class_planning = True
    cfg.hcp.max_number_classes = 5

    obstacles = pyteb.ObstacleContainer()
    obstacles.add(pyteb.CircularObstacle(2.5, 0.0, 0.5))

    planner = pyteb.HomotopyClassPlanner(cfg, obstacles)

    start = pyteb.PoseSE2(0.0, 0.0, 0.0)
    goal = pyteb.PoseSE2(5.0, 0.0, 0.0)

    def make_initial_plan():
        plan = [start]
        for i in range(1, 15):
            t = i / 15
            x = start.x + t * (goal.x - start.x)
            y = 1.2 * np.sin(np.pi * t)
            dy_dt = 1.2 * np.pi * np.cos(np.pi * t)
            dx_dt = goal.x - start.x
            theta = np.arctan2(dy_dt, dx_dt)
            plan.append(pyteb.PoseSE2(x, y, theta))
        plan.append(goal)
        return plan

    planner.plan(make_initial_plan(), pyteb.Velocity2D())

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'orange', 'purple', 'cyan', 'magenta']

    def update_plot():
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        trajectories = planner.get_trajectory_container()
        best_idx = planner.best_teb_idx()
        ax.set_title(f'Homotopy Interactive ({len(trajectories)} classes, click=add, r=replan, c=clear)')

        plot_obstacles(ax, obstacles)

        for i, teb_planner in enumerate(trajectories):
            teb = teb_planner.teb()
            poses = teb.get_all_poses()
            xs = [p.x for p in poses]
            ys = [p.y for p in poses]
            color = colors[i % len(colors)]
            alpha = 1.0 if i == best_idx else 0.3
            lw = 3 if i == best_idx else 1
            ax.plot(xs, ys, '-', color=color, linewidth=lw, alpha=alpha, label=f'Class {i+1}')

        plot_robot(ax, start, radius=0.15, color='green')
        plot_robot(ax, goal, radius=0.15, color='red')
        ax.legend(loc='upper right')
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            obstacles.add(pyteb.CircularObstacle(event.xdata, event.ydata, 0.2))
            print(f"Added obstacle at ({event.xdata:.2f}, {event.ydata:.2f})")
            planner.plan(make_initial_plan(), pyteb.Velocity2D())
            update_plot()

    def on_key(event):
        if event.key == 'r':
            print("Replanning...")
            planner.plan(make_initial_plan(), pyteb.Velocity2D())
            update_plot()
        elif event.key == 'c':
            print("Clearing obstacles...")
            obstacles.clear()
            planner.plan(make_initial_plan(), pyteb.Velocity2D())
            update_plot()
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TEB Planner Visualization')
    parser.add_argument('--mode', choices=['static', 'animated', 'interactive', 'homotopy', 'homotopy_interactive'],
                       default='static', help='Visualization mode')
    args = parser.parse_args()

    if args.mode == 'static':
        static_visualization()
    elif args.mode == 'animated':
        animated_visualization()
    elif args.mode == 'interactive':
        interactive_replanning()
    elif args.mode == 'homotopy':
        homotopy_visualization()
    elif args.mode == 'homotopy_interactive':
        interactive_homotopy()
