#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


from rclpy.executors import MultiThreadedExecutor
from  nav_msgs.msg import OccupancyGrid, MapMetaData
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np


class NoveltyMapping(Node):

    def __init__(self):
        super().__init__('Novelty_Node')
        self.cb_group = MutuallyExclusiveCallbackGroup()

        self.subscriber_ = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10, callback_group=self.cb_group)

        self.novelty_map_publisher = self.create_publisher(OccupancyGrid, '/novelty', 10)

        self.frontier_map_publisher = self.create_publisher(OccupancyGrid, '/frontier', 10)

        self.map = np.zeros((0,0))
        self.map_info: MapMetaData = None

        self.frontier_map = np.zeros((0,0))
        self.frontier_map_info: MapMetaData = None

        self.novelty_map = np.zeros((0,0))
        self.novelty_map_info: MapMetaData = None
        self.NUM_RAY_CASTS = 18
        self.angles = np.linspace(0, 2 * np.pi, self.NUM_RAY_CASTS, endpoint=False)
        self.cos_dirs = np.cos(self.angles)
        self.sin_dirs = np.sin(self.angles)


    def map_callback(self, map):
        array_map = np.asarray(map.data, dtype=np.int32).reshape(map.info.height, map.info.width)

        # Store safe numpy array map to self.map and MapMetaData as member variables
        self.map =  array_map
        self.map_info = map.info

        self.frontier_map = self.find_frontier(array_map)
        self.frontier_map_info = map.info

        frontier_grid = OccupancyGrid()
        frontier_grid.header = map.header
        frontier_grid.info = map.info
        frontier_grid.data = self.frontier_map.flatten().astype(np.int32).tolist()

        self.frontier_map_publisher.publish(frontier_grid)

        novelty_map = self.novelty_map_generator()
        self.novelty_map = novelty_map
        self.novelty_map_info = map.info

        novelty_grid = OccupancyGrid()
        novelty_grid.header = map.header
        novelty_grid.info = map.info
        novelty_grid.data = np.clip(novelty_map, 0, 100).flatten().astype(np.int8).tolist()

        self.novelty_map_publisher.publish(novelty_grid)


    def neighbors_of_8(self, p: tuple[int, int], map) -> list[tuple[int, int]]:
        """
        Returns the safe 8-neighbors cells of (x,y) in the occupancy grid.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable in 4 cardinal directions.
        """

        grid_x, grid_y = p

        # Create list of cells on all 4 sides of target
        neighbors = [
            (grid_x + 1, grid_y),
            (grid_x - 1, grid_y),
            (grid_x, grid_y + 1),
            (grid_x, grid_y - 1),
            (grid_x + 1, grid_y + 1),
            (grid_x - 1, grid_y - 1),
            (grid_x - 1, grid_y + 1),
            (grid_x + 1, grid_y - 1)
        ]

        safe_neighbors = []

        # Check each neighbor with the _is_safe function and add it to the list of safe neighbors
        for (x, y) in neighbors:
            if 0 <= x < self.map_info.height and 0 <= y < self.map_info.width:
                if map[x, y] == 0:
                    safe_neighbors.append((x, y))

        return safe_neighbors
    

    def find_frontier(self, map):
        unknown_cells = np.argwhere(map == -1)

        frontier_map = np.zeros(map.shape)
        


        for row, col in unknown_cells:
            if len(self.neighbors_of_8((row, col), map)) > 0:
                frontier_map[row, col] = 100

        return frontier_map


    def novelty_map_generator(self):

        novelty_grid = np.zeros(self.map.shape, dtype= np.int8)

        frontier_coords = np.argwhere(self.frontier_map)

        counter = 0

        divisor = int(len(frontier_coords)/50.0)

        if divisor == 0:
            divisor = 1

        for cord in frontier_coords:
            if counter % divisor == 0:
                visible_at_cord = self.visible_cells(cord)
                novelty_grid = novelty_grid + visible_at_cord

            counter += 1

        return novelty_grid


    def visible_cells(self, point: tuple[int, int]):
        MAX_RANGE = 1

        resolution = self.map_info.resolution
        max_range_cells = int(MAX_RANGE/resolution)
        
        
        EMPTY = 0
        UNKNOWN = 50
        OBSTACLE = 100

        grid = self.map

        rows, cols = grid.shape
        ox, oy = point

        if not (0 <= ox < rows and 0 <= oy < cols):
            raise ValueError(f"Origin {point} is outside grid bounds ({rows}, {cols})")
        if grid[ox, oy] == OBSTACLE:
            raise ValueError(f"Origin {point} is inside an obstacle")

        visible_map = np.zeros(grid.shape,dtype=np.int8)

        for i in range(self.NUM_RAY_CASTS):
            dx = self.cos_dirs[i]
            dy = self.sin_dirs[i]
            # Step along the ray using Bresenham-style sampling
            for step in range(1, max_range_cells + 1):
                rx = int(round(ox + step * dx))
                ry = int(round(oy + step * dy))

                # Stop if out of bounds
                if not (0 <= rx < rows and 0 <= ry < cols):
                    break

                cell = grid[rx, ry]

                if cell == OBSTACLE:
                    break  # Blocked — stop the ray, obstacle not counted
                elif cell == UNKNOWN and self.frontier_map[rx, ry] < 50:
                    break  # Can see the unknown cell but not past it
                elif cell == EMPTY:
                    visible_map[rx, ry] = 1

        return visible_map
    

def main(args=None):
    rclpy.init(args=args)

    novelty_mapping = NoveltyMapping()
    executor = MultiThreadedExecutor()
    executor.add_node(novelty_mapping)

    try:
        executor.spin()
    except KeyboardInterrupt:  # when 'ctrl + C' is pressed
        pass
    finally:
        novelty_mapping.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()