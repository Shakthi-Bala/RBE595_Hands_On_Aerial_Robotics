import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        #map4
        # self.start_point = [7.954360487979886, 6.822833826909669, 1.058209137433761]
        # self.goal_point = [44.304797815557095, 29.328280798754054, 4.454834705539382]
        #map1
        # self.start_point = [5.0,17.0,3.0]
        # self.goal_point = [3.954360487979886, -3.822833826909669, 4.058209137433761]
        #map2
        # self.start_point = [0,20,2] #start:  (0,20,2)
        # self.goal_point = [10,20,3]  #goal: (10, 20, 3)
        #map3
        self.start_point = [-0.5,3,2]  
        self.goal_point = [20, 2, 4] 
        self.safety_margin = 0.5 # Safety margin around obstacles
        


    def set_start_goal_points(self):
        return True 

    ###############################################
    ##### TODO - Implement map file parsing ####### 
    ###############################################    
    def parse_map_file(self, filename):
        """
        Parse the map file and extract boundary and blocks
        coords = [xmin, ymin, zmin, xmax, ymax, zmax]
        colors = [r, g, b] each in [0, 1] (make sure color values are in range 0-1)
        self.blocks.append((coords, colors))
        self.boundary = [xmin, ymin, zmin, xmax, ymax, zmax]
        return True if successful, False otherwise (True if file was parsed successfully, without any error.)
        """
    
        try:
            with open(filename,'r') as file:
                for line in file:
                    #remove whitespace and empty lines
                    clean_line=line.strip()
                    clean_line=" ".join(clean_line.split())

                    #skip comments and empty lines
                    if clean_line.startswith("#") or not clean_line:
                        continue
                    
                    #check if word boundary in line 
                    if "boundary" in clean_line:
                        parts=clean_line.split()
                        
                        values=[float(num) for num in parts[1:]]

                        if len(values)==6:
                            xmin, ymin, zmin, xmax, ymax, zmax=values
                            self.boundary=[xmin, ymin, zmin, xmax, ymax, zmax]
                    
                    #check if word block in line 
                    if "block" in clean_line:
                        parts=clean_line.split()
                        
                        values=[float(num) for num in parts[1:]]

                        if len(values)==9:
                            xmin, ymin, zmin, xmax, ymax, zmax, r, g, b=values
                            coords=[xmin, ymin, zmin, xmax, ymax, zmax]
                            #normalise rgb values within (0-1)
                            r, g, b=r/255, g/255, b/255
                            colors=[r, g, b]
                            self.blocks.append((coords,colors))

                    
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
            z = np.random.uniform(zmin + self.safety_margin, zmax - self.safety_margin)
            
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
        


#testing map parser 
# filename="/home/adityapat/RBE 595 Aerial Robotics/p2/p2_a/YourDirectoryID_p2a/src/maps/map1.txt"
# env=Environment3D()
# map_read_bool=env.parse_map_file(filename=filename)
# print(map_read_bool)
