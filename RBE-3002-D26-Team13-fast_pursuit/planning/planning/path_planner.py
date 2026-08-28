#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy.typing as npt

from rclpy.node import Node

import numpy as np
import math
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from  nav_msgs.msg import OccupancyGrid, Path, MapMetaData, GridCells, Odometry
from nav_msgs.srv import GetPlan
import cv2 as cv

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from priority_queue import PriorityQueue 

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

from custom_action_interfaces.action import PathGen
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

#-----------------------------------
#constants
ROBOT_RADIUS_M = 0.17/2           
 
# novelty_A_star cost weights
RISK_WEIGHT     = 3
DISTANCE_WEIGHT = 1
 
# find_basic_goal scoring weights
GOAL_NOVELTY_WEIGHT = 100.0
 
#---------------------------------------

class GraphNode():
    def __init__(self,pose,parent,cost):
        self.pose = pose
        self.parent = parent
        self.cost = cost
    
    def __eq__(self,other):
        return self.pose[0] == other.pose[0] and \
               self.pose[1] == other.pose[1]
    
    def __lt__(self,other):

        return id(self) < id(other)
    
    def __str__(self):
        return (f'{self.pose}')


class PathPlanner(Node):
    def __init__(self):
        # INDIVIDUAL 

        # TODO: Initialize the node and call it "path_planner"

        super().__init__('path_planner')
        self.cb_group = ReentrantCallbackGroup()
        self.private_cb_group = MutuallyExclusiveCallbackGroup()

        # TODO: Create Quality of Service (QoS) policy. Include a profile, depth, and durablilty policy. 
        # Hint: See the ROS 2 Docs if you get stuck!

        action_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability= ReliabilityPolicy.RELIABLE,
            durability = DurabilityPolicy.VOLATILE,
            )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability= ReliabilityPolicy.RELIABLE,
            durability = DurabilityPolicy.TRANSIENT_LOCAL,
            )

        #service
        self.create_service(GetPlan,'request_plan',self.plan_path, callback_group=self.private_cb_group)

        #subscriptions
        self.novelty_sub = self.create_subscription(OccupancyGrid, '/novelty', self.handle_novelty_map, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.handle_map, map_qos, callback_group=self.cb_group)
        # self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.handle_clicked_point, 10, callback_group=self.cb_group)

        #publishers
        self.safe_map_publisher = self.create_publisher(OccupancyGrid, '/map/safe', map_qos)
        self.risk_map_publisher = self.create_publisher(OccupancyGrid, '/risk', map_qos)
        self.visited_publisher = self.create_publisher(GridCells, '/path_planner/visited', 10)
        self.path_publisher = self.create_publisher(Path, '/nav_path', 10)

        

        # initial map values
        self.map : npt.NDArray[np.int32]         = np.zeros((0,0))
        self.risk_map : npt.NDArray[np.int32]    = np.zeros((0,0))
        self.novelty_map : npt.NDArray[np.int32] = np.zeros((0,0))
        self.map_info: Optional[MapMetaData]     = None

        #States
        self.goal : Tuple[int,int]              = None
        self.path_start_world : Optional[Point] = None
        self.goal_handle                        = None
        self.done                               = False
        self.replan_counter                     = 0
        self.plan: List[tuple[int,int]]         = None
        self.odom: Odometry                     = Odometry()

        # Setup TF Buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("path planner node initalized!")


    #------------------------------------------------------------------------------
    #Novelty Planner
    # ----------------------------------------------------------------------------- 

    def handle_novelty_map(self, map:OccupancyGrid):
        array_map = np.asarray(map.data, dtype=np.int32).reshape(map.info.height, map.info.width)
        safe_map = np.where(self.map > 50, 0, 1)
        if np.size(array_map) == np.size(safe_map):
            array_map = array_map*safe_map
            self.novelty_map = array_map


    def novelty_A_star(self, start : Tuple[int,int], goal: Tuple[int,int]):
        """
        Runs A* search from start coordinate to end coordinate using class member map.
        :param start    [[(int,int)]]     start coordinate in map    
        :param goal     [[(int,int)]]      goal coordinate in map    
        :return

        """

        self.get_logger().info(f"A* from {start} to {goal}")
        current_map     = self.map.copy()
        current_risk    = self.risk_map.copy()

        # Frontier initialized as a priority queue
        frontier = PriorityQueue()

        # First node saved as a graph node and placed in the frontier
        start_node = GraphNode(pose=start, parent=None, cost=0.0)
        start_heuristic = self.euclidean_distance(start, goal)
        frontier.put(start_node, start_heuristic)


        # List storing the best known cost to get to each cell
        best_cost: dict[tuple[int, int], float] = {start: 0.0}

        # List of cells that have been visited
        visited = []


        while not frontier.empty():
            # Gets the cell with the lowest cost from the frontier
            current = frontier.get()
            cell = current.pose[:2]

            if cell in visited:
                continue
            visited.append(cell)
                

            if cell == goal:
                # Visualization and debug
                self.get_logger().info(f"A* found path, visited {len(visited)} cells")
                self.draw_visited(visited)

                # Create the path and publish it to Rviz
                path = self.build_path(current)
                path_msg = self.build_path_message(path)
                self.path_publisher.publish(path_msg)
                return path
            

            for neighbor in self.neighbors_of_8(cell, current_map):
                if neighbor in visited or current_map[neighbor[1],neighbor[0]] == -1:
                    continue
                
                nx, ny = neighbor[0], neighbor[1]

                # Bounds check risk_map
                if ny >= current_risk.shape[0] or nx >= current_risk.shape[1]:
                    continue

                edge_distance = self.get_edge_cost(cell, neighbor)

                # The cost is calculated by combining the current cost, the cost to get to the neighbor, and the risk associated with the neighbor

                edge_cost = edge_distance * DISTANCE_WEIGHT
                risk_cost = current_risk[neighbor[1],neighbor[0]] * RISK_WEIGHT

                estimated_cost = edge_cost + risk_cost

                # If the cost is the best that has been seen save it to the frontier
                if estimated_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = estimated_cost
                    heuristic = self.euclidean_distance(neighbor, goal)
                    neighbor_node = GraphNode(pose=neighbor, parent=current, cost=estimated_cost)
                    frontier.put(neighbor_node, estimated_cost + heuristic)

        # Error if no path can be found
        self.get_logger().warn("A* could not find a path")
        return []           


    def build_path(self,final_node: GraphNode):
        """
        Given goal node from search tree, build path from start to goal to get there
        """

        current_node = final_node
        path = []
        while current_node is not None:
            path.append(current_node.pose)
            current_node= current_node.parent
        new_path = path[::-1]

        new_path = self.optimize_path(new_path)
        return new_path     

    #------------------------------------------------------
    # planning entry point
    #-----------------------------------------------------

    def plan_path(self, request: GetPlan.Request, response: GetPlan.Response) -> GetPlan.Response:
        """
        Plans a path between the current pose and the goal message locations.
        Internally uses A* to plan the optimal path.
        :param request  nav_msgs.srv._get_plan.GetPlan_Request  Start and End Pose for plan
        :param response nav_msgs.srv._get_plan.GetPlan_Response 
        :return         nav_msgs.srv._get_plan.GetPlan_Response     
        """
        self.get_logger().info("Service plan_path CALLED!")

        if self.map.size == 0 or self.map_info is None:
            self.get_logger().error("No map available")
            return response  # must return the response object, not []
        
        # Find cell index of start and goal poses in map
        start = (request.start.pose.position.x, request.start.pose.position.y)
        goal_grid   = self.snap_to_free(self.world_to_grid(self.map_info, request.goal.pose.position))
        goal_world = self.grid_to_world(self.map_info, goal_grid)
        

        # Find start and goal pose in grid
        start_point = Point(x= start[0], y = start[1], z = 0.0)
        start_grid = self.snap_to_free(self.world_to_grid(self.map_info, start_point))

        # Calculate a path using A* 
        path = self.novelty_A_star(start_grid, goal_grid)
        if not path:
            self.get_logger().warn('A* failed to find a path, returning empty plan.')
            return response
        
        self.path = path
        # Create path message
        new_plan = self.build_path_message(path)

        # Create a new pose for the world start pose
        start_pose = PoseStamped()
        start_pose.header = new_plan.header
        start_pose.pose.position.x = start[0]
        start_pose.pose.position.y = start[1]
        start_pose.pose.position.z = 0.0
        

        # add start pose to the beginning of path
        new_plan.poses.insert(0, start_pose)

        # Create a new pose for the world goal pose
        goal_pose = PoseStamped()
        goal_pose.header = new_plan.header
        goal_pose.pose.position.x = goal_world.x
        goal_pose.pose.position.y = goal_world.y
        goal_pose.pose.position.z = 0.0 

        # add goal pose to the end of the path
        new_plan.poses.append(goal_pose)
        
        response.plan.poses = new_plan.poses
        response.plan.header.frame_id = 'map'
        response.plan.header.stamp = self.get_clock().now().to_msg()
        return response

    
    #-----------------------------------------------------------------------
    # Map Callbacks
    #-----------------------------------------------------------------------

    def handle_map(self, map:OccupancyGrid):
        """
        Recieve raw map, convert to numpy array, expand obstacles.
        Save safe map and republish. 
        :param map      OccupancyGrid   The current map.
        """

        # self.get_logger().info("map received")

        # Convert map to 2D numpy array
        array_map = np.asarray(map.data, dtype=np.int32).reshape(map.info.height, map.info.width)

        # Number of grid cells to expand the map
        RES = map.info.resolution
        padding = int(math.ceil(ROBOT_RADIUS_M/RES)) 

        # Expand obstacles so your map represents where the robots center could be
        safe_area = self.risk_expansion(array_map, padding)

        # Store safe numpy array map to self.map and MapMetaData as member variables
        self.map = safe_area
        self.map_info = map.info

        # Create OccupancyGrid from safe numpy array map
        safe_grid = OccupancyGrid()
        safe_grid.header = map.header
        safe_grid.info = map.info
        safe_grid.data = safe_area.flatten().tolist()

        # Publish safe OccupancyGrid
        self.safe_map_publisher.publish(safe_grid)

        # Create the risk map
        self.create_risk_map()

        # Save risk map to OccupancyGrid and publish
        risk_grid = OccupancyGrid()
        risk_grid.header = map.header
        risk_grid.info = map.info
        risk_grid.data = self.risk_map.flatten().tolist()
        self.risk_map_publisher.publish(risk_grid)
        # self.get_logger().info('printing risk map')

        
    def create_risk_map(self):
        """This function creates a risk map based on distance from obstacles."""
        # creates empty map same size as base map
        current_map = self.map.copy()
        risk_map = np.zeros((self.map_info.height, self.map_info.width))
        max_dilation = 5   # was already computed, now actually used
        dilation = 1

        while dilation <= max_dilation:
            if 0 not in risk_map:
                break
            expanded = self.risk_expansion(current_map, dilation) / 50
            if expanded.shape != risk_map.shape:
                self.get_logger().warn(f'shape mismatch: {risk_map.shape} vs {expanded.shape}')
                break
            risk_map = risk_map + expanded
            dilation += 1

        self.risk_map = np.clip(risk_map.astype(np.int32), 0, 100)

    #---------------------------------------------------------------------------------------------
    # Map Helpers
    #---------------------------------------------------------------------------------------------

    @staticmethod
    def risk_expansion(original_map:npt.NDArray[np.int32], padding:int):
        """
        Expands obstacles to be "padding" larger in all directions
        :param original_map     ndarray     Map of obstacles
        :param padding          int         Number of cells from obstacle to consider unsafe
        :return                 ndarray     Map of safe configuration space.
        """

        # saves the map as a uint8 np array with 255 for obstacles and 0 everywhere else
        obstacle_img = np.uint8(original_map >= 50) * 255
        unknown_img  = np.uint8(original_map == -1) * 255

        obstacle_img = obstacle_img + unknown_img
        # set kernel for dilation
        kernel_size = 2 * padding + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Dilates the image by the specified padding
        dilated = cv.dilate(obstacle_img, kernel, iterations=1)

        # saves the safe_map as a int32 np array
        safe_map = np.where(dilated > 0, 100, 0).astype(np.int32)

        return safe_map
    
    @staticmethod
    def obstacle_expansion(original_map:npt.NDArray[np.int32], padding:int):
        """
        Expands obstacles to be "padding" larger in all directions
        :param original_map     ndarray     Map of obstacles
        :param padding          int         Number of cells from obstacle to consider unsafe
        :return                 ndarray     Map of safe configuration space.
        """

        # saves the map as a uint8 np array with 255 for obstacles and 0 everywhere else
        obstacle_img = np.uint8(original_map >= 50) * 255

        # set kernel for dilation
        kernel_size = 2 * padding + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Dilates the image by the specified padding
        dilated = cv.dilate(obstacle_img, kernel, iterations=1)

        # saves the safe_map as a int32 np array
        safe_map = np.where(dilated > 0, 100, 0).astype(np.int32)

        return safe_map




#---------------------------------------------------------------------------------------------
# Coordinate Transforms
#---------------------------------------------------------------------------------------------
    @staticmethod
    def grid_to_world(mapdata: MapMetaData, p: tuple[int, int]) -> Point:
        """
        Transforms a cell coordinate in the occupancy grid into a world coordinate.
        :param mapdata [MapMetaData] The map information.
        :param p [(int, int)] The cell coordinate.
        :return        [Point]         The position in the world.
        """

        res = mapdata.resolution

        origin_x = mapdata.origin.position.x
        origin_y = mapdata.origin.position.y

        world_point = Point()
        world_point.x = float(origin_x + (p[0] + 0.5) * res)
        world_point.y = float(origin_y + (p[1] + 0.5) * res)
        
        return world_point


        
    @staticmethod
    def world_to_grid(mapdata: MapMetaData, wp: Point) -> tuple[int, int]:
        """
        Transforms a world coordinate into a cell coordinate in the occupancy grid.
        :param mapdata [MapMetaData] The map information.
        :param wp      [Point]         The world coordinate.
        :return        [(int,int)]     The cell position as a tuple.
        """

        res = mapdata.resolution

        origin_x = mapdata.origin.position.x
        origin_y = mapdata.origin.position.y

        grid_x = int((wp.x - origin_x) / res)
        grid_y = int((wp.y - origin_y) / res)

        return (grid_x, grid_y)


    @staticmethod
    def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """
        Calculates the Euclidean distance between two points.
        :param p1 [(float, float)] first point.
        :param p2 [(float, float)] second point.
        :return   [float]          distance.
        """

        return math.dist(p1, p2)
        

    def build_path_message(self, path: list[tuple[int, int]]) -> Path:
        """
        Converts a list of cell coordinates to a Path Message
        :param path     [(int,int)]     The cell coordinates corresponding to the current map
        :return         Path            
        """

        # Create empty path and set header
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for cell in path:
            # Convert grid cordinates into world
            world_point = self.grid_to_world(self.map_info, cell)

            # Create empty pose and set header
            pose = PoseStamped()
            pose.header = path_msg.header

            # Set the pose.position to the position in world coordinates
            pose.pose.position = world_point
            
            # Add the new pose to the path at the end
            path_msg.poses.append(pose)

        # Return the generated path
        return path_msg


    def _is_safe(self, grid_x: int, grid_y: int, map = None) -> bool:

        # Get the size of the map
        if map is None:
            map = self.map
        height, width = map.shape

        # check if point is inside of the grid bounds
        if grid_x < 0 or grid_y < 0 or grid_x >= width or grid_y >= height:
            return False
        
        # Return False if the cell has a value over 50 indicating that it is an obstacle return True otherwise
        return map[grid_y, grid_x] < 50


    def neighbors_of_4(self,p: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Returns the safe 4-neighbors cells of (x,y) in the occupancy grid.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable in 4 cardinal directions.
        """

        grid_x, grid_y = p

        # Create list of cells on all 4 sides of target
        neighbors = [
            (grid_x + 1, grid_y),
            (grid_x - 1, grid_y),
            (grid_x, grid_y + 1),
            (grid_x, grid_y - 1)
        ]

        safe_neighbors = []

        # Check each neighbor with the _is_safe function and add it to the list of safe neighbors
        for (x, y) in neighbors:
            if self._is_safe(x, y):
                safe_neighbors.append((x, y))


        return safe_neighbors

    
    def neighbors_of_8(self, p: tuple[int, int], map = None) -> list[tuple[int, int]]:
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
            if self._is_safe(x, y, map):
                safe_neighbors.append((x, y))


        return safe_neighbors


    def get_edge_cost(self,p1: tuple[int, int],p2: tuple[int, int]):
        """
        Compute cost to traverse between 2 nodes.
        :param p1       [(int, int)]    The coordinate in the grid of start.
        :param p2       [(int, int)]    The coordinate in the grid of end.

        """
        
        return self.euclidean_distance(p1, p2)
    

    def draw_visited(self,visited):
        """
        Draw set of visited node as GridCells message.
        """
        cells = GridCells()
        cells.cell_width = self.map_info.resolution
        cells.cell_height = self.map_info.resolution
        cells.header.stamp = self.get_clock().now().to_msg()
        cells.header.frame_id = 'map'
        for index in visited:
            p = self.grid_to_world(self.map_info,index[:2])            
            cells.cells.append(p)
        self.visited_publisher.publish(cells)


    

    def optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Optimizes the path, removing unnecessary intermediate nodes.
        :param path [[(x,y)]] The path as a list of tuples (grid coordinates)
        :return     [[(x,y)]] The optimized path as a list of tuples (grid coordinates)
        """   
        if len(path) < 3:
            return path

        i = 0
        while True:
            if i + 2 == len(path): 
                return path

            # Checks 3 points at a time and if the three points are colinear the middle one is deleted
            if self.is_collinear(path[i], path[i + 1], path[i + 2]):
                # self.get_logger().info(f'removing {path[i+1]}')
                del path[i+1]
            
            else:
                i += 1


    def is_collinear(self,p1, p2, p3, epsilon=1e-12):
        """
        Checks if three points are collinear using the cross product.
        p1, p2, p3 are tuples/lists of (x, y) coordinates.
        """

        # Vector 1: p2 - p1
        x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
        # Vector 2: p3 - p1
        x2, y2 = p3[0] - p1[0], p3[1] - p1[1]

        return abs(x1 * y2 - x2 * y1) < epsilon

    def neighbors_with_orientation(self,p: tuple[int, int,int]) -> list[tuple[int, int, int]]:
        """
        Returns the safe neighbours for nonholonomic robot with pose x,y,theta.
        :param mapdata [OccupancyGrid] The map information.
        :param p       [(int, int, int)]    The coordinate in the grid.
        :return        [[(int,int, int)]]   A list of walkable 8-neighbors.
        """
        # EXTRA CREDIT      
        pass
    
    def snap_to_free(self, cell: tuple[int,int], radius: int = 5) -> tuple[int,int]:
        """Return nearest free cell within `radius` of `cell`, or cell itself."""
        cx, cy = cell
        for r in range(0, radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # only check the ring at distance r
                    nx, ny = cx + dx, cy + dy
                    if self._is_safe(nx, ny):
                        return (nx, ny)
        return cell  # couldn't find one, return original


def main(args=None):
    rclpy.init(args=args)
    # TODO: add your node
    planner = PathPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(planner)
    try:
        # TODO: Spin your node
        executor.spin()

    except KeyboardInterrupt:
        pass
    finally:
        # TODO: Destroy and shutdown node when ctrl + C is pressed
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
