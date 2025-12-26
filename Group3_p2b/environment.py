import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        self.start_point = [0, 0, -0.1]
        self.goal_point = [1.6, 0.125, -0.1]
        self.safety_margin = 0.12  # Safety margin around obstacles



    ###############################################
    ##### TODO - Implement map file parsing ####### 
    ###############################################    
    def parse_map_file(self, filename):
        """
        Parse map file into boundary and blocks, tolerating commas and stray dashes.
        """
        try:
            self.boundary = []
            self.blocks = []

            with open(filename, 'r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Normalize separators: turn commas into spaces, collapse whitespace
                    line = line.replace(',', ' ')
                    line = ' '.join(line.split())

                    # Split and filter obvious junk tokens like a lone '-'
                    parts = [p for p in line.split(' ') if p and p != '-']
                    if not parts:
                        continue

                    key = parts[0].lower()

                    # BOUNDARY: expect 6 numbers
                    if key == 'boundary':
                        nums = []
                        for tok in parts[1:]:
                            try:
                                nums.append(float(tok))
                            except ValueError:
                                # skip any leftover junk tokens
                                continue
                        if len(nums) < 6:
                            raise ValueError(f'Boundary needs 6 numbers, got {len(nums)} at line: {raw.strip()}')
                        xmin, ymin, zmin, xmax, ymax, zmax = nums[:6]
                        # ensure min<max
                        xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
                        ymin, ymax = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
                        zmin, zmax = (zmin, zmax) if zmin <= zmax else (zmax, zmin)
                        self.boundary = [xmin, ymin, zmin, xmax, ymax, zmax]
                        continue

                    # BLOCK: accept either 6 (coords) or 9 (coords+rgb)
                    if key == 'block':
                        nums = []
                        for tok in parts[1:]:
                            try:
                                nums.append(float(tok))
                            except ValueError:
                                continue

                        if len(nums) not in (6, 9):
                            raise ValueError(f'Block needs 6 or 9 numbers, got {len(nums)} at line: {raw.strip()}')

                        xmin, ymin, zmin, xmax, ymax, zmax = nums[:6]
                        xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
                        ymin, ymax = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
                        zmin, zmax = (zmin, zmax) if zmin <= zmax else (zmax, zmin)
                        coords = [xmin, ymin, zmin, xmax, ymax, zmax]

                        if len(nums) == 9:
                            r, g, b = nums[6:9]
                        else:
                            # default color if none provided
                            r, g, b = 200.0, 200.0, 200.0

                        # clamp and normalize 0–255 -> 0–1
                        r = max(0.0, min(255.0, r)) / 255.0
                        g = max(0.0, min(255.0, g)) / 255.0
                        b = max(0.0, min(255.0, b)) / 255.0
                        self.blocks.append((coords, [r, g, b]))
                        continue

                    # Unknown keywords silently ignored
                    # print(f"Warning: Unknown line ignored: {raw.strip()}")

            if not self.boundary:
                raise ValueError('No boundary specified in map file.')

            return True

        except FileNotFoundError:
            print(f"Error: {filename} was not found ")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False


    ##############################################
    #### TODO - Implement collision checking #####
    ##############################################
    def is_point_in_free_space(self, point):
        """
        Check if a point is in free space (not inside any obstacle)
        Complete implementation with collision checking
        return True if free, False if in collision
        """

        # Using min and max 
        
        #boundary check 
        x_min=self.boundary[0]
        y_min=self.boundary[1]
        z_min=self.boundary[2]
        x_max=self.boundary[3]
        y_max=self.boundary[4]
        z_max=self.boundary[5]
        
        # boundary check 
        if not (point[0]>=x_min and point[0]<=x_max and point[1]>=y_min and point[1]<=y_max and point[2]>=z_min and point[2]<=z_max):
            return False


        #block check + safety margin 
        for block in self.blocks:
            x_min=block[0][0]-self.safety_margin
            y_min=block[0][1]-self.safety_margin
            z_min=block[0][2]-self.safety_margin
            x_max=block[0][3]+self.safety_margin
            y_max=block[0][4]+self.safety_margin
            z_max=block[0][5]+self.safety_margin

            if point[0]>=x_min and point[0]<=x_max and point[1]>=y_min and point[1]<=y_max and point[2]>=z_min and point[2]<=z_max:
                return False

        else:
            return True
    


    ##############################################
    #### TODO - Implement line - collision checking #####
    ##############################################
    def is_line_collision_free(self, p1, p2, num_checks=20):
        """
        Check if a line segment between two points is collision-free
        Used for RRT* edge validation
        return True if free, False if in collision
        """
        p1=self.standardize_point(p1)
        p2=self.standardize_point(p2)
        # generate points along line segment of (p1,p2) check inidividually if points are colliding with boundary/blocks
        line_seg_points=[]
        # print("p1:",p1)
        # print("p2:",p2)

        # print("p1:",p1[0][0])
        x_points=np.linspace(p1[0],p2[0],num_checks)
        y_points=np.linspace(p1[1],p2[1],num_checks)
        z_points=np.linspace(p1[2],p2[2],num_checks)
        for x,y,z in zip(x_points, y_points, z_points)  :
            point = [x, y, z]
            if not self.is_point_in_free_space(point):
                return False # Collision found
            
        


        return True
    

    
    def generate_random_free_point(self):
        """
        Generate a random point in free space
        Used for RRT* sampling
        """
        if not self.boundary:
            return None
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(xmin + self.safety_margin, xmax - self.safety_margin)
            y = np.random.uniform(ymin + self.safety_margin, ymax - self.safety_margin)
            # z = np.random.uniform(zmin + self.safety_margin, zmax - self.safety_margin)
            z=-0.1
            
            point = [x, y, z]
            
            if self.is_point_in_free_space(point):
                return point
        
        print("Warning: Could not generate random free point after", max_attempts, "attempts")
        return None

    def get_environment_info(self):
        """Get information about the environment layout"""
        if not self.boundary:
            return "No boundary defined"
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        info = f"""
                Environment Information:
                Boundary: [{xmin}, {ymin}, {zmin}] to [{xmax}, {ymax}, {zmax}]
                Size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} meters
                Volume: {(xmax-xmin)*(ymax-ymin)*(zmax-zmin):.1f} cubic meters
                Obstacles: {len(self.blocks)} blocks
                Safety margin: {self.safety_margin} meters
                """
        
        if self.start_point and self.goal_point:
            distance = np.linalg.norm(np.array(self.goal_point) - np.array(self.start_point))
            info += f"  Start-Goal distance: {distance:.2f} meters\n"
        
        return info

    def get_environment_info(self):
        """Get information about the environment layout"""
        if not self.boundary:
            return "No boundary defined"
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        info = f"""
                Environment Information:
                Boundary: [{xmin}, {ymin}, {zmin}] to [{xmax}, {ymax}, {zmax}]
                Size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} meters
                Volume: {(xmax-xmin)*(ymax-ymin)*(zmax-zmin):.1f} cubic meters
                Obstacles: {len(self.blocks)} blocks
                Safety margin: {self.safety_margin} meters
                """
        
        if self.start_point and self.goal_point:
            distance = np.linalg.norm(np.array(self.goal_point) - np.array(self.start_point))
            info += f"  Start-Goal distance: {distance:.2f} meters\n"
        
        return info

    # Helper function to standardize point format
    def standardize_point(self,point_data):
        """
        Normalizes a point's data structure into a simple 1D NumPy array.

        This function handles two common formats:
        1. A list/tuple containing a single NumPy array, e.g., [array([x, y, z])]
        2. A simple list, tuple, or 1D array of numbers, e.g., [x, y, z]

        Args:
            point_data: The point data in one of the two supported formats.

        Returns:
            A simple 1D NumPy array of the point's coordinates.
        """
        # Check if the input is a list or tuple that contains a single NumPy array.
        # This is the signature of your nested format: [array([...])]
        if isinstance(point_data, (list, tuple)) and len(point_data) == 1 and isinstance(point_data[0], np.ndarray):
            # If it matches, we just need the inner array.
            return point_data[0]
        else:
            # Otherwise, we assume it's already a flat structure like [7.95, 6.82, 1.05].
            # We convert it to a NumPy array to ensure the output type is always consistent.
            return np.array(point_data)
        

    
    def set_start_goal_points(self, start=None, goal=None):
        """Set start/goal. If not given, sample collision-free points."""
        # Use provided or defaults
        if start is None:
            start = self.start_point
        if goal is None:
            goal = self.goal_point

        # If defaults are None or invalid, sample
        def valid(pt): return pt is not None and self.is_point_in_free_space(pt)

        if not valid(start):
            rs = self.generate_random_free_point()
            if rs is None: return False
            self.start_point = rs
        else:
            self.start_point = list(start)

        if not valid(goal):
            rg = self.generate_random_free_point()
            if rg is None: return False
            self.goal_point = rg
        else:
            self.goal_point = list(goal)

        # Avoid identical points
        if np.allclose(self.start_point, self.goal_point, atol=1e-3):
            rg = self.generate_random_free_point()
            if rg is None: return False
            self.goal_point = rg

        return True

    def visualize_environment(self, ax=None, show_start_goal=True):
        """Quick 3D plot of boundary, blocks, and (optionally) start/goal."""
        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False

        # Draw boundary edges
        if self.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
            verts = [
                [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i,j in edges:
                ax.plot([verts[i][0], verts[j][0]],
                        [verts[i][1], verts[j][1]],
                        [verts[i][2], verts[j][2]], 'k--', alpha=0.3, linewidth=1)

        # Draw blocks
        for (xmin, ymin, zmin, xmax, ymax, zmax), color in self.blocks:
            xs = [xmin, xmax]; ys = [ymin, ymax]; zs = [zmin, zmax]
            X, Y = np.meshgrid(xs, ys)
            ax.plot_surface(X, Y, np.full_like(X, zs[0]), alpha=0.2, color=color)
            ax.plot_surface(X, Y, np.full_like(X, zs[1]), alpha=0.2, color=color)

        # Start/Goal
        if show_start_goal:
            if self.start_point: ax.scatter(*self.start_point, c='g', s=120, marker='s', edgecolors='k', label='Start')
            if self.goal_point:  ax.scatter(*self.goal_point,  c='gold', s=150, marker='*', edgecolors='k', label='Goal')

        if self.boundary:
            xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title('Environment')
        if show_start_goal: ax.legend()
        if standalone: plt.tight_layout(); plt.show()
        return ax