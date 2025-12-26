import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt



class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using  
splines
    Complete implementation with velocity and acceleration profiles
    """
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        # self.waypoints= [self.waypoints[0]]*2 + self.waypoints + [self.waypoints[-1]]*2
        # map splat configuration
        self.trajectory_duration = 10  # seconds
        self.max_velocity = 1.0 # m/s
        self.max_acceleration = 1  # m/s^2
        self.average_velocity = 0.5   # m/s
        self.total_duration = 0.0


        if len(waypoints) > 3:
            self.tck=None
        else:
            self.tck=None
            print("Not enough waypoints for spline fitting. Need at least 4 points.")

    ##############################################################
    #### TODO - Implement spline trajectory generation ###########
    #### TODO - Ensure velocity and acceleration constraints #####
    #### TODO - Add member functions as needed ###################
    ##############################################################
    def generate_bspline_trajectory(self, num_points=None):
        """
        Generate spline trajectory with complete velocity/acceleration profiles
        """

        trajectory_points = None
        time_points = None
        velocities = None
        accelerations = None
        ramp_duration=4.0  # seconds

        dt=0.02

        ############## IMPLEMENTATION STARTS HERE ##############

        # if self.tck is None:
        #     print("Error: Spline coefficients (tck) not available.")
        #     return None, None, None, None
        
        distances = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        segment_times = distances / self.average_velocity
    
        time_knots = np.zeros(len(self.waypoints))
        time_knots[1:] = np.cumsum(segment_times)
        self.total_duration = time_knots[-1]


        # u_fine= np.linspace(0 ,1, num_points)
        # time_points = np.linspace(0, self.trajectory_duration, num_points)

        self.tck, _ = splprep(self.waypoints.T, u=time_knots, k=3, s=0)

        num_points = int(np.ceil(self.total_duration / dt))
        time_points = np.linspace(0, self.total_duration, num_points)

        x_coords, y_coords, z_coords = splev(time_points, self.tck)
        trajectory_points = np.vstack((x_coords, y_coords, z_coords)).T

        #First derivative for velocity
        dp_du=splev(time_points, self.tck, der=1)
        velocities=np.vstack(dp_du).T/ self.trajectory_duration

        #Second derivative for acceleration
        d2p_du2=splev(time_points, self.tck, der=2)
        accelerations=np.vstack(d2p_du2).T/ (self.trajectory_duration**2)

        dt = self.trajectory_duration / (num_points - 1) if num_points > 1 else 0
        # ramp_steps = int(ramp_duration / dt)

        # # ramp up profile
        # if ramp_steps > 0 and ramp_steps < num_points:
        #     ramp_profile = np.linspace(0, 1, ramp_steps)

        #     ramp_profile = 0.5 * (1 - np.cos(np.pi * ramp_profile)) 


        #     velocities[:ramp_steps] *= ramp_profile.reshape(-1, 1)
        #     accelerations[:ramp_steps] *= ramp_profile.reshape(-1, 1)


            
        return trajectory_points, time_points, velocities, accelerations
            

 
    def visualize_trajectory(self, trajectory_points=None, velocities=None, 
                           accelerations=None, ax=None):
        """Visualize the trajectory with velocity and acceleration vectors"""
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            standalone = True
        else:
            ax1 = ax
            standalone = False
        
        if trajectory_points is not None:
            # Plot 3D trajectory
            ax1.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    trajectory_points[:, 2], 'b-', linewidth=2, label='Spline Trajectory')
            
            # Plot waypoints
            ax1.plot(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], 
                    'ro-', markersize=8, linewidth=2, label='Waypoints')
            
            # Plot velocity vectors (sampled)
            if velocities is not None:
                step = max(1, len(trajectory_points) // 20)  # Show ~20 vectors
                for i in range(0, len(trajectory_points), step):
                    pos = trajectory_points[i]
                    vel = velocities[i] * 0.5  # Scale for visualization
                    ax1.quiver(pos[0], pos[1], pos[2], 
                             vel[0], vel[1], vel[2], 
                             color='green', alpha=0.7, arrow_length_ratio=0.1)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()
        
        if standalone and velocities is not None and accelerations is not None:
            # Plot velocity magnitude over time
            time_points = np.linspace(0, self.trajectory_duration, len(velocities))
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            ax2.plot(time_points, vel_magnitudes, 'g-', linewidth=2)
            ax2.axhline(y=self.max_velocity, color='r', linestyle='--', 
                       label=f'Max Vel: {self.max_velocity} m/s')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Profile')
            ax2.grid(True)
            ax2.legend()
            
            # Plot acceleration magnitude over time
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            ax3.plot(time_points, acc_magnitudes, 'm-', linewidth=2)
            ax3.axhline(y=self.max_acceleration, color='r', linestyle='--', 
                       label=f'Max Acc: {self.max_acceleration} m/s²')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Acceleration (m/s²)')
            ax3.set_title('Acceleration Profile')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
        
        return ax1 if not standalone else None
    
    # def generate_bspline_trajectory():

    #     pass