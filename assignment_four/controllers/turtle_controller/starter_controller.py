"""student_controller controller."""

import math
import heapq
import numpy as np
from collections import deque

LEFT = -2.75
RIGHT = 2.75
TOP = 2.75
BOTTOM = -2.75

BIG_OBSTACLE_SIZE = 0.5
SMALL_OBSTACLE_SIZE = 0.25

GRID_LENGTH = 0.03125

# grow all obstacles by robot radius + small safety margin
ROBOT_RADIUS = 0.09
SAFETY_MARGIN = 0.04
INFLATION = 0.10

# A* preferences
CLEARANCE_RADIUS_CELLS = 5
CLEARANCE_PENALTY = 8.0

# path following
WAYPOINT_REACHED_DIST = 0.08
ANGLE_TOL = 0.05
TURN_IN_PLACE_THRESH = 0.12
FORWARD_SPEED = 1.8
FAST_FORWARD_SPEED = 2.5
MAX_TURN_SPEED = 3.0
KP_TURN = 3.0

# keep all corner waypoints
PATH_RESOLUTION = 1

class StudentController:
   def __init__(self):
       self.rows = int((TOP - BOTTOM) / GRID_LENGTH)
       self.cols = int((RIGHT - LEFT) / GRID_LENGTH)


       self.occupancy = None
       self.clearance_map = None
       self.path_cells = None
       self.path_world = None
       self.waypoint_idx = 0
       self.initialized = False
       self.last_goal = None

   # conversions from world to grid space and vice versa
   def world_to_grid(self, x, y):
       col = int((x - LEFT) / GRID_LENGTH)
       row = int((TOP - y) / GRID_LENGTH)


       row = max(0, min(self.rows - 1, row))
       col = max(0, min(self.cols - 1, col))
       return (row, col)


   def grid_to_world(self, row, col):
       x = LEFT + (col + 0.5) * GRID_LENGTH
       y = TOP - (row + 0.5) * GRID_LENGTH
       return (x, y)


   def normalize_angle(self, angle):
       while angle > math.pi:
           angle -= 2 * math.pi
       while angle < -math.pi:
           angle += 2 * math.pi
       return angle


   # config space stuff
   def point_in_grown_rect(self, x, y, cx, cy, half_size):
       return (cx - half_size <= x <= cx + half_size) and (cy - half_size <= y <= cy + half_size)


   def build_occupancy(self, map_data, obstacles):
       grid_array = np.array(map_data)


       if grid_array.ndim == 1:
           side_length = int(math.sqrt(grid_array.size))
           coarse_map = grid_array.reshape((side_length, side_length))
       else:
           coarse_map = grid_array


       occ = np.zeros((self.rows, self.cols), dtype=np.uint8)


       # build list of grown rectangles for walls + small obstacles
       grown_rects = []


       # big wall blocks from map
       for i in range(coarse_map.shape[0]):
           for j in range(coarse_map.shape[1]):
               if coarse_map[i, j] != 1:
                   continue

               cx = LEFT + BIG_OBSTACLE_SIZE / 2 + BIG_OBSTACLE_SIZE * j
               cy = TOP - BIG_OBSTACLE_SIZE / 2 - BIG_OBSTACLE_SIZE * i
               half_size = BIG_OBSTACLE_SIZE / 2 + INFLATION
               grown_rects.append((cx, cy, half_size))


       # small obstacles
       for obs in obstacles:
           cx, cy = obs[0], obs[1]
           half_size = SMALL_OBSTACLE_SIZE / 2 + INFLATION
           grown_rects.append((cx, cy, half_size))


       # put grown obstacles into fine occupancy grid
       for r in range(self.rows):
           for c in range(self.cols):
               x, y = self.grid_to_world(r, c)

               blocked = False
               for cx, cy, half_size in grown_rects:
                   if self.point_in_grown_rect(x, y, cx, cy, half_size):
                       blocked = True
                       break

               if blocked:
                   occ[r, c] = 1


       return occ

   # clearance map
   def build_clearance_map(self, occupancy):
       dist = np.full((self.rows, self.cols), np.inf, dtype=float)
       q = deque()


       occupied = np.argwhere(occupancy == 1)
       for r, c in occupied:
           dist[r, c] = 0
           q.append((r, c))


       directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


       while q:
           r, c = q.popleft()
           for dr, dc in directions:
               nr = r + dr
               nc = c + dc
               if 0 <= nr < self.rows and 0 <= nc < self.cols:
                   if dist[nr, nc] > dist[r, c] + 1:
                       dist[nr, nc] = dist[r, c] + 1
                       q.append((nr, nc))


       return dist


   def nearest_free_cell(self, cell, max_radius=20):
       r0, c0 = cell
       if self.occupancy[r0, c0] == 0:
           return cell


       for radius in range(1, max_radius + 1):
           for dr in range(-radius, radius + 1):
               for dc in range(-radius, radius + 1):
                   r = r0 + dr
                   c = c0 + dc
                   if 0 <= r < self.rows and 0 <= c < self.cols:
                       if self.occupancy[r, c] == 0:
                           return (r, c)


       return cell


   # A* implmentation and helpers
   def heuristic(self, a, b):
       # euclidean heuristic
       return math.hypot(a[0] - b[0], a[1] - b[1])


   def get_neighbors(self, node):
       r, c = node
       neighbors = []


       directions = [
           (-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1)
       ]

       for dr, dc in directions:
           nr = r + dr
           nc = c + dc

           if not (0 <= nr < self.rows and 0 <= nc < self.cols):
               continue

           if self.occupancy[nr, nc] == 1:
               continue

           # prevent diagonal corner cutting
           if dr != 0 and dc != 0:
               if self.occupancy[r + dr, c] == 1 or self.occupancy[r, c + dc] == 1:
                   continue

           neighbors.append((nr, nc))

       return neighbors


   def step_cost(self, a, b):
       # base movement cost
       base = math.sqrt(2) if (a[0] != b[0] and a[1] != b[1]) else 1.0


       # add soft penalty for being near obstacles
       clearance = self.clearance_map[b[0], b[1]]
       penalty = 0.0
       if clearance < CLEARANCE_RADIUS_CELLS:
           penalty = CLEARANCE_PENALTY * (CLEARANCE_RADIUS_CELLS - clearance)


       return base + penalty


   def reconstruct_path(self, came_from, current):
       path = [current]
       while current in came_from:
           current = came_from[current]
           path.append(current)
       path.reverse()
       return path


   def astar(self, start, goal):
       open_heap = []
       counter = 0
       heapq.heappush(open_heap, (0.0, counter, start))


       came_from = {}
       g_score = {start: 0.0}
       f_score = {start: self.heuristic(start, goal)}
       closed = set()


       while open_heap:
           _, _, current = heapq.heappop(open_heap)

           if current in closed:
               continue
           
           if current == goal:
               return self.reconstruct_path(came_from, current)

           closed.add(current)

           for neighbor in self.get_neighbors(current):
               if neighbor in closed:
                   continue
               
               tentative_g = g_score[current] + self.step_cost(current, neighbor)

               if neighbor not in g_score or tentative_g < g_score[neighbor]:
                   came_from[neighbor] = current
                   g_score[neighbor] = tentative_g
                   f = tentative_g + self.heuristic(neighbor, goal)
                   f_score[neighbor] = f
                   counter += 1
                   heapq.heappush(open_heap, (f, counter, neighbor))

       return None

   # path post process
   def simplify_path(self, path):
       if path is None or len(path) < 3:
           return path

       simplified = [path[0]]
       prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])

       for i in range(2, len(path)):
           curr_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
           if curr_dir != prev_dir:
               simplified.append(path[i - 1])
           prev_dir = curr_dir

       simplified.append(path[-1])
       return simplified

   def plan_path(self, pose, goal):
       start = self.world_to_grid(pose[0], pose[1])
       goal_cell = self.world_to_grid(goal[0], goal[1])


       start = self.nearest_free_cell(start)
       goal_cell = self.nearest_free_cell(goal_cell)


       raw_path = self.astar(start, goal_cell)
       if raw_path is None:
           return None, None


       simple_path = self.simplify_path(raw_path)
       world_path = [self.grid_to_world(r, c) for (r, c) in simple_path]
       return simple_path, world_path


   # controller
   def step(self, sensors):
       control_dict = {"left_motor": 0.0, "right_motor": 0.0}

       pose = sensors["pose"]
       map_data = sensors["map"]
       goal = sensors["goal"]
       obstacles = sensors["obstacles"]

       goal_tuple = (goal[0], goal[1])

       # build occupancy if goal changes
       if (not self.initialized) or (self.last_goal != goal_tuple):
           self.occupancy = self.build_occupancy(map_data, obstacles)
           self.clearance_map = self.build_clearance_map(self.occupancy)
           self.initialized = True
           self.last_goal = goal_tuple

       # replan every step from current pose
       self.path_cells, self.path_world = self.plan_path(pose, goal)

       if self.path_world is None or len(self.path_world) == 0:
           return control_dict

       self.waypoint_idx = 1 if len(self.path_world) > 1 else 0

       target_x, target_y = self.path_world[self.waypoint_idx]

       dx = target_x - pose[0]
       dy = target_y - pose[1]
       dist = math.hypot(dx, dy)

       target_theta = math.atan2(dy, dx)
       angle_diff = self.normalize_angle(target_theta - pose[2])

       # if very close to this waypoint and another exists aim at the next one
       if dist < WAYPOINT_REACHED_DIST and len(self.path_world) > self.waypoint_idx + 1:
           self.waypoint_idx += 1
           target_x, target_y = self.path_world[self.waypoint_idx]
           dx = target_x - pose[0]
           dy = target_y - pose[1]
           dist = math.hypot(dx, dy)
           target_theta = math.atan2(dy, dx)
           angle_diff = self.normalize_angle(target_theta - pose[2])


       # if abs(angle_diff) > 0.5:
       #     turn_speed = MAX_TURN_SPEED
       #     if angle_diff > 0:
       #         control_dict["left_motor"] = -turn_speed
       #         control_dict["right_motor"] = turn_speed
       #     else:
       #         control_dict["left_motor"] = turn_speed
       #         control_dict["right_motor"] = -turn_speed
       #     return control_dict


       # smooth controller P based controller
       turn = KP_TURN * angle_diff

       # clip turn
       turn = np.clip(turn, -MAX_TURN_SPEED, MAX_TURN_SPEED)


       # reduce forward speed when turning
       forward = FAST_FORWARD_SPEED * max(0.0, 1.0 - abs(angle_diff))

       # enforce minimum forward speed so it doesn't stall
       forward = max(forward, 1.2)

       control_dict["left_motor"] = forward - turn
       control_dict["right_motor"] = forward + turn

       return control_dict