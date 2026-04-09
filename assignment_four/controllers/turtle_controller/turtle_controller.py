import math
import numpy as np
from controller import Robot, DistanceSensor, Motor, Compass, GPS
from controller import Supervisor
from starter_controller import StudentController


MIN_VELOCITY = -6.25
MAX_VELOCITY = 6.25
MAX_BOXES = 10

# TEST = 2


# Define the robot class
class TurtleBotController:
    def __init__(self):
        # Initialize the robot and get basic time step
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Create the Supervisor instance
        # self.supervisor = Supervisor()
        self.robot_node = self.robot.getFromDef("MY_ROBOT")
        self.box_nodes = [
            self.robot.getFromDef(f"OBSTACLE_{d}").getField("translation").getSFVec3f()[:2] for d in range(100) if self.robot.getFromDef(f"OBSTACLE_{d}") is not None
        ]

        # print('obstacles', self.box_nodes)

        # Get motors (assuming left and right motors are available)
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")

        # Set motor velocity to maximum (adjust as needed)
        self.left_motor.setPosition(float("inf"))  # Infinity means continuous rotation
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Initialize Lidar (Laser Distance Sensor)
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)

        # Initialize Compass (for orientation)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)

        # Sensor and control noise
        self._lidar_noise = 0.15
        self._detection_range = 1.0
        self._control_noise_pct = 0.05
        self._heading_noise = 0 * (math.pi / 180)

        self.grid_map = np.load("gridmap.npy")
        self.goal = (2, 2)
        # if TEST == 1:
        #     self.grid_map = np.load("test_one.npy")
        #     self.goal = (-1, 2)
        #     # self.goal = (2, 4)
        # elif TEST == 2:
        #     self.grid_map = np.load("test_two.npy")
        #     self.goal = (-2, 1)
        #     # self.goal = (4, 2)
        self.student_controller = StudentController()

    def provide_lidar(self):
        """Get lidar range image and lower detect range below default 3.5."""
        lidar_image = self.lidar.getRangeImage()
        for i in range(len(lidar_image)):
            lidar_image[i] = (
                float("inf")
                if lidar_image[i] >= self._detection_range
                else lidar_image[i]
            )
        noise = np.random.normal(0, self._lidar_noise, size=len(lidar_image))
        lidar_image += noise
        return lidar_image

    def provide_compass(self):
        values = self.compass.getValues()
        heading = math.atan2(values[0], values[1])
        # noise = np.random.normal(0, self._heading_noise)
        return heading  # + noise

    def clip_control(self, control):
        """Non-linear behavior in controls."""
        if abs(control) < 0.05:
            control = 0.0
        control = min(control, max(control, MIN_VELOCITY), MAX_VELOCITY)
        return control

    def provide_pose(self):
        position = self.robot_node.getField("translation").getSFVec3f()[:2]
        rotation = self.provide_compass()
        return position + [rotation]

    def run(self):
        """
        The main loop that controls the robot's behavior.
        """
        print(self.grid_map)

        while self.robot.step(self.time_step) != -1:
            # Pack sensor values for student controller
            sensors = {
                "pose": self.provide_pose(),  # [x, y, theta] in meters and radians.
                "map": self.grid_map,  # 2D numpy array where 1 is a wall.
                "goal": self.goal,  # coordinates to navigate to.
                "obstacles": self.box_nodes,  # each obstacle is a 0.25 x 0.25m box aligned with the grid.
            }

            # Get control values from student controller
            controls = self.student_controller.step(sensors)
            lwhl = controls.get("left_motor", 0.0)
            rwhl = controls.get("right_motor", 0.0)

            # Apply noise to control and clip to remain in bounds
            lwhl += np.random.normal(0, self._control_noise_pct * abs(lwhl))
            rwhl += np.random.normal(0, self._control_noise_pct * abs(rwhl))
            lwhl = self.clip_control(lwhl)
            rwhl = self.clip_control(rwhl)

            # Set control
            self.left_motor.setVelocity(lwhl)
            self.right_motor.setVelocity(rwhl)


# Create a controller instance and run it
controller = TurtleBotController()

controller.run()
