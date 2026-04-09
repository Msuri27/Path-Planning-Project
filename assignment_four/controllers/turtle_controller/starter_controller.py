"""student_controller controller."""

import math
import numpy as np


class StudentController:
    def __init__(self):
        pass

    # HELPER METHODS:

    # create new map with scale factor of 6 for full res.
    def create_c_space(map, obstacles, cell_resolution=0.0833):
        
        grid_array = np.array(map)
    
        # calculate side length
        side_length = int(math.sqrt(grid_array.size))
        grid_2d = grid_array.reshape((side_length, side_length))

        # scale matrix up by 6
        scale_factor = 6
        c_space = np.repeat(np.repeat(grid_2d, scale_factor, axis=0), scale_factor, axis=1)

        maze_width_m = 5.5
        maze_height_m = 5.5

        # for relative coords
        x_min = -(maze_width_m / 2.0)
        y_max = (maze_height_m / 2.0)

        # 3 cells of padding from center on all sides (7x7)
        pad = 3

        for (obs_x, obs_y) in obstacles:
            # offset origin to bottom left corner
            dist_x  = obs_x - x_min
            dist_y = y_max - obs_y

            # calculate index
            col = np.floor(dist_x / cell_resolution)
            row = np.floor(dist_y / cell_resolution)

            # define 7x7 bounding box and clip for out of bounds errors
            r_start = max(0, row - pad)
            r_end = min(c_space.shape[0], col + pad +1)
            c_start = max(0, col - pad)
            c_end = min(c_space.shape[1], col + pad + 1)

            # assign obstacle grid cells 1
            c_space[r_start:r_end, c_start:c_end] = 1

        return c_space

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # TODO: add your controllers here.
        control_dict["left_motor"] = 6.0
        control_dict["right_motor"] = 6.0

        # unpack sensors
        pose = sensors["pose"]
        map = sensors["map"]
        goal = sensors["goal"]
        obstacles = sensors["obstacles"]

        print(type(map))

        c_space = self.create_c_space(map, obstacles)    
        print(c_space)    

        return control_dict
