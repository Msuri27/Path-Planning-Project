import argparse
import random
import matplotlib.pyplot as plt
import numpy as np

# To generate a new maze:
# 1. First, run this script to generate a new "gridmap.npy" and "maze_boxes.wbt"
# 2. Second, copy the contents of maze_boxes.wbt into turtle_world.wbt.
# 3. Set the desired goal position in turtle_controller.py (in the __init__ function).


def generate_maze(width, height):
    maze = [[1] * (2 * width + 1) for _ in range(2 * height + 1)]

    def carve_passages(cx, cy):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and maze[2 * ny + 1][2 * nx + 1] == 1
            ):
                maze[2 * cy + 1 + dy][2 * cx + 1 + dx] = 0
                maze[2 * ny + 1][2 * nx + 1] = 0
                carve_passages(nx, ny)

    maze[1][1] = 0  # Start position
    carve_passages(0, 0)

    return maze


def display_maze(maze):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap="binary")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def print_boxes(maze, obstacle_prob=0.25, basename="maze"):
    start_x = -2.5
    start_y = -2.5
    obstacle_count = 0
    header = """\
#VRML_SIM R2023b utf8
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/robotis/turtlebot/protos/TurtleBot3Burger.proto"
EXTERNPROTO "https://github.com/cyberbotics/webots/blob/released/projects/objects/shapes/protos/TexturedBoxShape.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/solids/protos/SolidBox.proto"
# EXTERNPROTO "/Users/jphanna/teaching/25spring-cs639/turtle_project/protos/MyProto.proto"

WorldInfo {
}
Viewpoint {
  orientation -0.5773 0.5773 0.5773 2.0944
  position 0 0 10
}
TexturedBackground {
}
TexturedBackgroundLight {
}
RectangleArena {
  floorSize 5 5
  wallHeight 0.1
}
DEF MY_ROBOT TurtleBot3Burger {
  name "myRobot"
  controller "turtle_controller"
  translation -2 -2 0
  rotation 0 0 1 0
  supervisor TRUE
}
"""
    with open(f"{basename}.wbt", 'w') as w:
        w.write(header)
        # Write wall boxes
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == 0:
                    continue
                x = j * 0.5 + start_x
                y = i * 0.5 + start_y
                w.write("SolidBox {\n")
                w.write('    name "' + f"wall_{i * len(maze[i]) + j}" + '"\n')
                w.write(f"    translation {x} {y} 0" + "\n")
                w.write("    size 0.5 0.5 1\n}\n")

        # Scatter obstacles in open (passable) cells, flush against an adjacent wall
        # Cell half-width=0.25, obstacle half-width=0.125 → flush offset = 0.25-0.125 = 0.125
        flush_offset = 0.125
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] != 0:
                    continue
                if random.random() > obstacle_prob:
                    continue
                # Find which of the 4 neighbours are wall cells
                # (di, dj): di → world-y offset, dj → world-x offset
                wall_dirs = [
                    (di, dj)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= i + di < len(maze)
                    and 0 <= j + dj < len(maze[i])
                    and maze[i + di][j + dj] == 1
                ]
                if not wall_dirs:
                    continue  # open cell with no adjacent walls — skip
                di, dj = random.choice(wall_dirs)
                cx = j * 0.5 + start_x
                cy = i * 0.5 + start_y
                ox = cx + dj * flush_offset
                oy = cy + di * flush_offset
                w.write("DEF OBSTACLE_" + str(obstacle_count) + " SolidBox {\n")
                w.write('    name "' + f"obstacle_{obstacle_count}" + '"\n')
                w.write(f"    translation {ox:.4f} {oy:.4f} 0\n")
                w.write("    size 0.25 0.25 1\n}\n")
                obstacle_count += 1

    print(f"Placed {obstacle_count} obstacles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random maze.")
    parser.add_argument(
        "--basename",
        default="maze",
        help="Base file name (no extension) for the output .wbt and .npy files (default: maze)",
    )
    args = parser.parse_args()

    width, height = 5, 5  # Define maze size
    maze = generate_maze(width, height)
    # display_maze(np.array(maze))
    print_boxes(maze, obstacle_prob=0.5, basename=args.basename)
    np.save(f"{args.basename}.npy", np.flipud(np.array(maze)))
