import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp
import time
import os
import io
import cv2
from pyquaternion import Quaternion

# Local imports
from environment import Environment3D
from path_planner import PathPlanner, RRTNode
from trajectory_generator import TrajectoryGenerator
from control import QuadrotorController

# Dynamics and parameters
from quad_dynamics import model_derivative
import tello as drone_params

# splat rendering
from splat_render import SplatRenderer


class LiveQuadrotorSimulator:
    """
    Real-time live visualization quadrotor simulator
    Phases: RRT* planning -> B-spline -> Execution (+ video recording)
    """

    def __init__(self, map_file=None):
        # Initialize environment
        self.env = Environment3D()
        if map_file:
            success = self.env.parse_map_file(map_file)
            if not success:
                print(f"Failed to load map file: {map_file}")

        self.planner = PathPlanner(self.env)
        self.traj_gen = None
        self.controller = QuadrotorController(drone_params)

        # Renderer config
        self.splatConfig = '/home/alien/YourDirectoryID_p2b/p2phaseb_colmap_splat/p2phaseb_colmap/splatfacto/2025-10-07_134702/config.yml'
        self.renderSettings = '/home/alien/YourDirectoryID_p2b/p2b.json'
        self.renderer = SplatRenderer(self.splatConfig, self.renderSettings)

        # Logs
        os.makedirs('./log', exist_ok=True)

        # Simulation parameters (keep in sync with controller)
        self.dt = 0.02  # 50 Hz
        self.sim_time = 0.0
        self.max_sim_time = 30.0

        # State: [x,y,z, vx,vy,vz, qx,qy,qz,qw, p,q,r]
        self.state = np.zeros(13)
        self.state[9] = 1.0  # quaternion w

        # Histories
        self.state_history = []
        self.time_history = []
        self.control_history = []

        # Splat output directory (optional)
        self.render_dir = '/home/alien/YourDirectoryID_p2b/renders'
        os.makedirs(self.render_dir, exist_ok=True)

        # Status flags
        self.goal_reached = False
        self.goal_tolerance = 0.03  # meters; slightly relaxed for stability
        self.simulation_active = False

        # Visualization handles
        self.fig = None
        self.ax = None
        self.drone_point = None
        self.drone_trail = None
        self.trail_positions = []
        self.max_trail_length = 300

        # RRT* viz handles
        self.rrt_tree_lines = []
        self.rrt_nodes_scatter = None
        self.rrt_path_line = None
        self.bspline_line = None

        # Phase flags
        self.planning_complete = False
        self.trajectory_complete = False
        self.execution_started = False

        # Video recording
        self.video_writer = None
        self.video_path = os.path.join(os.getcwd(), "Video.mp4")
        self.record_video = True
        self._video_fourcc = None
        self._video_fps = None
        self.z_coordinate= 0
        # Default render size (SplatRenderer ~1080p height with 4:3 AR)
        self.splat_h = 1080
        self.splat_w = int(1080 * (4 / 3))

    # ---------- Visualization setup ----------

    def setup_visualization(self):
        """Setup the 3D visualization for step-by-step planning"""
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self._draw_environment()

        # Set axes limits if boundary exists
        if self.env.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Quadrotor Path Planning and Execution')

        plt.show(block=False)

    # ---------- Optional environment preview ----------

    def preview_environment(self, wait_time=3.0):
        print("\n" + "=" * 60)
        print("👀 ENVIRONMENT PREVIEW")
        print("=" * 60)
        print(self.env.get_environment_info())
        self.ax.set_title('🌍 Environment Preview - Inspecting workspace...')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if wait_time is None:
            input("\n▶️  Press ENTER to start planning...")
        else:
            print(f"\n⏱️  Displaying environment for {wait_time:.1f} seconds...")
            for i in range(int(wait_time), 0, -1):
                print(f"   Starting planning in {i}...", end='\r')
                time.sleep(1)
            print("\n▶️  Starting planning now!            ")
        print("=" * 60 + "\n")

    # ---------- RRT* planning ----------

    def animated_rrt_planning(self, start=None, goal=None):
        """Show animated RRT* planning process"""
        print("🎬 Starting animated RRT* planning...")

        # Respect provided start/goal if given
        if start is not None:
            self.env.start_point = list(start)
        if goal is not None:
            self.env.goal_point = list(goal)

        print(self.env.get_environment_info())

        start_point = self.env.start_point
        goal_point = self.env.goal_point
        distance = np.linalg.norm(np.array(goal_point) - np.array(start_point))

        print(f"📏 Start-to-goal distance: {distance:.2f} meters")
        if distance < 0.10:
            print("⚠️  Warning: Start and goal are quite close")
        elif distance > 15.0:
            print("⚠️  Warning: Start and goal are quite far - this may take longer")
        else:
            print("✅ Good start-goal separation for planning")

        self.ax.set_title('Phase 1: RRT* Path Planning (Building Tree...)')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        start_node = RRTNode(start_point)
        tree = [start_node]

        # Scale iterations with distance; ensure 1000–3000 bounds
        max_iterations = min(3000, max(1000, int(distance * 150)))
        step_size = max(0.1, min(1.2, distance / 12))  # slightly gentler than before
        goal_radius = max(0.5, min(1.2, distance / 16))
        search_radius = max(0.3, step_size * 2.2)
        goal_bias = 0.15

        print("🔧 RRT* Parameters:")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Step size: {step_size:.3f} m")
        print(f"   Goal radius: {goal_radius:.3f} m")
        print(f"   Search radius: {search_radius:.3f} m")

        goal_node = None
        update_interval = max(25, max_iterations // 80)

        print(f"🚀 Starting RRT* planning from {start_point} to {goal_point}")

        for iteration in range(max_iterations):
            # Sample
            if np.random.random() < goal_bias:
                sample_point = np.array(goal_point)
            else:
                sample = self.env.generate_random_free_point()
                if sample is None:
                    continue
                sample_point = np.array(sample)

            # Nearest in tree
            nearest_node = self.planner.find_nearest_node(tree, sample_point)
            if nearest_node is None:
                continue

            # Steer towards sample
            new_position = self.planner.steer(nearest_node.position, sample_point, step_size)

            # Checks
            if not self.env.is_point_in_free_space(new_position):
                continue
            if not self.env.is_line_collision_free(nearest_node.position, new_position):
                continue

            # Parent choice among nearby nodes
            near_nodes = self.planner.find_near_nodes(tree, new_position, search_radius)
            best_parent, best_cost = self.planner.choose_parent(near_nodes, new_position)

            if best_parent is None:
                if self.planner.is_path_valid(nearest_node.position, new_position):
                    best_parent = nearest_node
                    best_cost = nearest_node.cost + self.planner.distance(nearest_node.position, new_position)
                else:
                    continue

            # Create node
            new_node = RRTNode(new_position)
            new_node.parent = best_parent
            new_node.cost = best_cost
            best_parent.children.append(new_node)
            tree.append(new_node)

            # Rewire
            self.planner.rewire_tree(tree, new_node, near_nodes)

            # Check goal
            goal_distance = self.planner.distance(new_position, goal_point)
            if goal_distance <= goal_radius and self.planner.is_path_valid(new_position, goal_point):
                goal_node = RRTNode(goal_point)
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + goal_distance
                new_node.children.append(goal_node)
                tree.append(goal_node)
                print(f"🏁 Goal reached at iteration {iteration}! Final cost: {goal_node.cost:.2f}")
                break

            # Viz update
            if iteration % update_interval == 0:
                self._update_rrt_visualization(tree, iteration, max_iterations)
                time.sleep(0.03)

        # Final viz update
        self._update_rrt_visualization(tree, iteration, max_iterations, final=True)

        # Store results
        self.planner.tree_nodes = tree

        if goal_node is not None:
            self.planner.waypoints = self.planner.extract_path(goal_node)
            original_waypoints = len(self.planner.waypoints)
            self.planner.waypoints = self.planner.simplify_path(self.planner.waypoints)
            simplified_waypoints = len(self.planner.waypoints)

            print("✅ RRT* planning successful!")
            print(f"   Original path: {original_waypoints} waypoints")
            print(f"   Simplified path: {simplified_waypoints} waypoints")
            print(f"   Path cost: {goal_node.cost:.2f} meters")
            print(f"   Tree size: {len(tree)} nodes")

            self._show_final_rrt_path()
            return True

        print(f"❌ RRT* planning failed after {max_iterations} iterations")
        print(f"   Tree size: {len(tree)} nodes")
        print("   Try increasing max_iterations or adjusting parameters")
        return False

    def _update_rrt_visualization(self, tree, iteration, max_iterations, final=False):
        """Update RRT* tree visualization"""
        for line in self.rrt_tree_lines:
            line.remove()
        self.rrt_tree_lines = []

        if self.rrt_nodes_scatter is not None:
            self.rrt_nodes_scatter.remove()
            self.rrt_nodes_scatter = None

        # Draw edges (sample for perf)
        sample_rate = max(1, len(tree) // 500)
        for node in tree[::sample_rate]:
            if node.parent is not None:
                line, = self.ax.plot(
                    [node.parent.position[0], node.position[0]],
                    [node.parent.position[1], node.position[1]],
                    [node.parent.position[2], node.position[2]],
                    'b-', alpha=0.3, linewidth=0.5
                )
                self.rrt_tree_lines.append(line)

        # Draw nodes (sample)
        if len(tree) > 1:
            sampled_nodes = tree[::max(1, len(tree) // 200)]
            positions = np.array([n.position for n in sampled_nodes])
            self.rrt_nodes_scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c='blue', s=8, alpha=0.6
            )

        progress = (iteration / max_iterations) * 100
        status = "COMPLETE" if final else f"{progress:.1f}%"
        self.ax.set_title(f'Phase 1: RRT* Planning - {status} (Nodes: {len(tree)}, Iter: {iteration})')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _show_final_rrt_path(self):
        time.sleep(1)
        if len(self.planner.waypoints) > 0:
            waypoints = np.array(self.planner.waypoints)
            self.rrt_path_line, = self.ax.plot(
                waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                'ro-', markersize=8, linewidth=4, label='RRT* Path', alpha=0.9
            )
        self.ax.set_title('Phase 1: RRT* Planning - COMPLETE! Final path shown.')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(2)
        self.planning_complete = True

    # ---------- B-spline trajectory ----------

    def show_bspline_trajectory(self):
        print("📈 Generating B-spline trajectory...")

        self.ax.set_title('Phase 2: B-spline Trajectory Generation...')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.traj_gen = TrajectoryGenerator(self.planner.waypoints)
        # Let the generator compute its own duration from average_velocity if set;
        # otherwise use a conservative cap so it’s not rushed.
        if not getattr(self.traj_gen, "trajectory_duration", None):
            self.traj_gen.trajectory_duration = min(25.0, max(8.0, len(self.planner.waypoints) * 1.8))

        num_points = int(self.traj_gen.trajectory_duration / self.dt)
        result = self.traj_gen.generate_bspline_trajectory(num_points=num_points)

        if result[0] is None:
            print("❌ Trajectory generation failed")
            return False

        trajectory_points, time_points, velocities, accelerations = result
        self.controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)
        self.max_sim_time = self.traj_gen.trajectory_duration + 5.0

        # Draw trajectory (sample)
        traj_sample = trajectory_points[::5]
        self.bspline_line, = self.ax.plot(
            traj_sample[:, 0], traj_sample[:, 1], traj_sample[:, 2],
            'g-', linewidth=3, alpha=0.8, label='B-spline Trajectory'
        )

        # Add some velocity vectors
        vector_sample = max(1, len(trajectory_points) // 20)
        for i in range(0, len(trajectory_points), vector_sample):
            pos = trajectory_points[i]
            vel = velocities[i] * 0.4
            self.ax.quiver(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                           color='orange', alpha=0.7, arrow_length_ratio=0.1)

        self.ax.set_title('Phase 2: B-spline Trajectory - COMPLETE!')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        time.sleep(2)
        self.trajectory_complete = True
        return True

    # ---------- Execution ----------

    def initialize_execution_phase(self):
        print("🚁 Starting execution phase...")

        # Initial state
        self.state[0:3] = self.env.start_point
        self.state[3:6] = 0.0
        self.state[6:10] = [0, 0, 0, 1]
        self.state[10:13] = 0.0

        # Drone marker
        current_pos = self.state[0:3]
        self.drone_point = self.ax.scatter(
            *current_pos, c='red', s=200, marker='o',
            edgecolors='black', linewidths=2, label='Quadrotor'
        )

        # Trail
        self.trail_positions = [current_pos.copy()]
        self.drone_trail, = self.ax.plot(
            [current_pos[0]], [current_pos[1]], [current_pos[2]],
            'purple', linewidth=4, alpha=0.9, label='Executed Path'
        )

        # Metrics
        self.controller.reset_metrics()
        self.sim_time = 0.0
        self.goal_reached = False

        self.ax.set_title('Phase 3: Executing Trajectory...')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Prep video writer (lazy-open when first frame is ready)
        if self.record_video:
            self._video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_fps = int(round(1.0 / self.dt))
            self.video_writer = None

        self.execution_started = True

    def _fig_to_rgb(self):
        """Return current Matplotlib figure as an RGB uint8 array, backend-proof and size-stable."""
        # Draw and capture a PNG of the figure to avoid backend differences
        self.fig.canvas.draw()
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02)
        buf.seek(0)

        png_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)  # decode PNG -> BGR
        if bgr is None:
            # ultra fallback, try without tight bbox
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png")
            buf.seek(0)
            png_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("Failed to capture Matplotlib frame to PNG")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _capture_and_write_frame(self):
        """Render splat camera + matplotlib, hstack, and write to MP4."""
        if not self.record_video:
            return

        # 1) SPLAT view from current pose (use actual attitude)
        x, y, z = self.state[0:3]
        self.z_coordinate-=10e-4
        z+=self.z_coordinate
        # z=-0.1
        qx, qy, qz, qw = self.state[6:10]
        quat = Quaternion([qw, qx, qy, qz])
        yaw, pitch, roll = quat.yaw_pitch_roll
        rpy = np.array([0.0, 0.0, yaw], dtype=float)


        try:
            rgb_splat, depth_splat = self.renderer.render(
            position=np.array([x, y, z], dtype=float),
            orientation_rpy=rpy
            )
        except Exception:
            return  # skip this frame if renderer hiccups

        # 2) Matplotlib view of world
        rgb_mpl = self._fig_to_rgb()

        # 3) Resize Mpl to Splat height, keep aspect
        h_s, w_s, _ = rgb_splat.shape
        h_m, w_m, _ = rgb_mpl.shape
        new_w_m = int(w_m * (h_s / float(h_m)))
        rgb_mpl_resized = cv2.resize(rgb_mpl, (new_w_m, h_s), interpolation=cv2.INTER_AREA)

        # 4) Stack and write
        side_by_side = np.hstack([rgb_splat, rgb_mpl_resized])  # RGB

        if self.video_writer is None:
            H, W, _ = side_by_side.shape
            self.video_writer = cv2.VideoWriter(
                self.video_path, self._video_fourcc, self._video_fps, (W, H)
            )
            if not self.video_writer.isOpened():
                print("❌ Could not open VideoWriter for", self.video_path)
                self.record_video = False
                return

        bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        self.video_writer.write(bgr)

    def simulation_step(self):
        """Single simulation time step"""
        if not self.simulation_active:
            return False

        # Control (50 Hz)
        control_input = self.controller.compute_control(self.state, self.sim_time)

        # Dynamics
        def dynamics(t, x):
            return model_derivative(
                t, x.reshape(-1, 1),
                control_input.reshape(-1, 1),
                drone_params
            ).flatten()

        sol = solve_ivp(dynamics, [self.sim_time, self.sim_time + self.dt],
                        self.state, method='RK45')

        self.state = sol.y[:, -1]
        self.sim_time += self.dt

        # Logs
        self.state_history.append(self.state.copy())
        self.time_history.append(self.sim_time)
        self.control_history.append(control_input.copy())

        # Trail
        current_pos = self.state[0:3]
        self.trail_positions.append(current_pos.copy())
        if len(self.trail_positions) > self.max_trail_length:
            self.trail_positions.pop(0)

        # Goal check
        if self.env.goal_point is not None:
            goal_distance = np.linalg.norm(current_pos - np.array(self.env.goal_point))
            if goal_distance < self.goal_tolerance and not self.goal_reached:
                self.goal_reached = True
                print(f"\n🎉 GOAL REACHED at time {self.sim_time:.2f}s!")
                print(f"📏 Final distance to goal: {goal_distance:.3f} m")
                return False

        # Time limit
        if self.sim_time >= self.max_sim_time:
            print(f"\n⏰ Simulation time limit reached: {self.sim_time:.2f}s")
            return False

        return True

    def update_execution_visualization(self):
        """Update the execution visualization"""
        if self.drone_point is None:
            return

        current_pos = self.state[0:3]

        # Update drone position
        self.drone_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])

        # Update trail
        if len(self.trail_positions) > 1:
            trail_array = np.array(self.trail_positions)
            self.drone_trail.set_data_3d(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2])

        # Title with info
        goal_dist = "N/A"
        if self.env.goal_point is not None:
            goal_dist = f"{np.linalg.norm(current_pos - np.array(self.env.goal_point)):.2f} m"

        self.ax.set_title(f'Phase 3: Executing - Time: {self.sim_time:.1f}s, Goal Dist: {goal_dist}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run_live_simulation(self, start=None, goal=None):
        """
        Run the complete simulation:
        1. RRT* planning (animated)
        2. B-spline generation
        3. Execution (with video capture)
        """
        print("🎬 Starting complete quadrotor simulation with all phases...")

        self.setup_visualization()

        # Phase 1
        if not self.animated_rrt_planning(start, goal):
            return False

        # Phase 2
        if not self.show_bspline_trajectory():
            return False

        # Phase 3
        self.initialize_execution_phase()

        self.simulation_active = True
        print("\n🚀 Executing trajectory...")
        print("💡 Close the plot window to stop simulation")

        # Visualization FPS (decoupled from control dt)
        execution_update_rate = 25  # Hz
        update_interval = 1.0 / execution_update_rate
        last_update_time = time.time()

        try:
            while self.simulation_active and plt.get_fignums():
                start_loop = time.time()

                continue_sim = self.simulation_step()

                # Visualization + frame capture at viz rate
                now = time.time()
                if now - last_update_time >= update_interval:
                    self.update_execution_visualization()
                    self._capture_and_write_frame()
                    last_update_time = now

                # Keep real-time pace
                elapsed = time.time() - start_loop
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if not continue_sim:
                    break

        except KeyboardInterrupt:
            print("\n⏹️  Simulation stopped by user")
        except Exception as e:
            print(f"\n💥 Simulation error: {e}")
        finally:
            self.simulation_active = False
            # Final capture and close video
            self._capture_and_write_frame()
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print(f"🎥 Saved video to {self.video_path}")

        # Final viz update and summary
        self.update_execution_visualization()
        self._print_simulation_results()

        print("\n📚 Simulation complete. Close plot window to continue...")
        plt.ioff()
        plt.show()
        return True

    # ---------- Environment drawing ----------

    def _draw_environment(self):
        """Draw the static environment elements"""
        # Boundary wireframe
        if self.env.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
            vertices = self._create_cube_vertices(xmin, ymin, zmin, xmax, ymax, zmax)
            faces = self._create_cube_faces(vertices)
            for face in faces:
                fa = np.array(face + [face[0]])
                self.ax.plot(fa[:, 0], fa[:, 1], fa[:, 2], 'k--', alpha=0.3, linewidth=1)

        # Obstacles
        for block_coords, block_color in self.env.blocks:
            vertices = self._create_cube_vertices(*block_coords)
            faces = self._create_cube_faces(vertices)
            poly3d = [[tuple(v) for v in face] for face in faces]
            self.ax.add_collection3d(Poly3DCollection(
                poly3d, facecolors=block_color, alpha=0.8, edgecolors='black', linewidths=0.5
            ))

        # Start & goal
        if self.env.start_point:
            self.ax.scatter(*self.env.start_point, c='green', s=150, marker='s',
                            edgecolors='black', linewidth=2, label='Start')
        if self.env.goal_point:
            self.ax.scatter(*self.env.goal_point, c='gold', s=150, marker='*',
                            edgecolors='black', linewidth=2, label='Goal')

    def _create_cube_vertices(self, xmin, ymin, zmin, xmax, ymax, zmax):
        """8 vertices of an axis-aligned box."""
        return [
            np.array([xmin, ymin, zmin]), np.array([xmax, ymin, zmin]),
            np.array([xmax, ymax, zmin]), np.array([xmin, ymax, zmin]),
            np.array([xmin, ymin, zmax]), np.array([xmax, ymin, zmax]),
            np.array([xmax, ymax, zmax]), np.array([xmin, ymax, zmax]),
        ]

    def _create_cube_faces(self, vertices):
        """Faces from vertices."""
        return [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
            [vertices[1], vertices[5], vertices[6], vertices[2]],  # right
            [vertices[4], vertices[0], vertices[3], vertices[7]],  # left
        ]

    # ---------- Offline init (optional) ----------

    def initialize_simulation(self, start=None, goal=None):
        """Prep everything for an offline run (no live window)."""
        if not self.env.set_start_goal_points(start, goal):
            print("Failed to set start/goal points")
            return False

        planner = PathPlanner(self.env)
        if not planner.plan_path():
            print("Path planning failed")
            return False
        self.planner = planner

        self.traj_gen = TrajectoryGenerator(self.planner.waypoints)
        result = self.traj_gen.generate_bspline_trajectory(
            num_points=int(max(200, self.traj_gen.trajectory_duration / self.dt))
        )
        if result[0] is None:
            print("Trajectory generation failed")
            return False

        traj_pts, t_pts, vels, accs = result
        self.controller.set_trajectory(traj_pts, t_pts, vels, accs)
        self.max_sim_time = self.traj_gen.trajectory_duration + 5.0

        self.state[0:3] = self.env.start_point
        self.state[3:6] = 0.0
        self.state[6:10] = [1, 0, 0, 0]
        self.state[10:13] = 0.0
        self.sim_time = 0.0
        self.goal_reached = False
        self.simulation_active = True
        self.trail_positions = [self.state[0:3].copy()]
        return True

    # ---------- Results & saving ----------

    def _print_simulation_results(self):
        print("\n" + "=" * 60)
        print("🏁 COMPLETE SIMULATION RESULTS")
        print("=" * 60)

        final_pos = self.state[0:3]

        print("📊 PHASE SUMMARY:")
        print(f"   ✅ Phase 1: RRT* Planning - {len(self.planner.waypoints)} waypoints")
        print(f"   ✅ Phase 2: B-spline Trajectory - "
              f"{len(self.controller.trajectory_points) if self.controller.trajectory_points is not None else 0} points")
        print(f"   ✅ Phase 3: Execution - {self.sim_time:.2f}s")

        if self.env.goal_point is not None:
            goal_distance = np.linalg.norm(final_pos - np.array(self.env.goal_point))
            start_goal_dist = np.linalg.norm(np.array(self.env.goal_point) - np.array(self.env.start_point))
            success_rate = max(0, (1 - goal_distance / start_goal_dist) * 100)
            print("\n🏅 EXECUTION RESULTS:")
            print(f"   Status: {'✅ GOAL REACHED' if self.goal_reached else '❌ NOT REACHED'}")
            print(f"   Final distance to goal: {goal_distance:.3f} m")
            print(f"   Success rate: {success_rate:.1f}%")

        print("\n📈 PERFORMANCE:")
        print(f"   Execution time: {self.sim_time:.2f} s")
        print(f"   Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
        print(f"   Path length: {len(self.trail_positions)} points")

        if self.controller.position_errors:
            mean_pos_error = np.mean(self.controller.position_errors)
            max_pos_error = np.max(self.controller.position_errors)
            print(f"   Tracking error: {mean_pos_error:.3f} m avg, {max_pos_error:.3f} m max")

        print("=" * 60)

    def save_results(self, filename_prefix='complete_simulation'):
        import scipy.io
        data = {
            'time': np.array(self.time_history),
            'states': np.array(self.state_history),
            'controls': np.array(self.control_history),
            'rrt_waypoints': np.array(self.planner.waypoints),
            'bspline_trajectory': self.controller.trajectory_points,
            'executed_trail': np.array(self.trail_positions),
            'goal_reached': self.goal_reached,
            'sim_time': self.sim_time,
            'start_point': np.array(self.env.start_point),
            'goal_point': np.array(self.env.goal_point)
        }
        filename = f'./log/{filename_prefix}.mat'
        scipy.io.savemat(filename, data)
        print(f"💾 Complete simulation results saved to {filename}")
