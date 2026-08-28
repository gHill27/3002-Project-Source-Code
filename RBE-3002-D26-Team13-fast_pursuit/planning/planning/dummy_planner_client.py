#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

import numpy as np
from typing import Tuple, List
from nav_msgs.msg import Path, Odometry, OccupancyGrid, MapMetaData
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import cv2 as cv
import heapq

from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from rclpy.action import ActionClient
from custom_action_interfaces.action import PathGen
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
import math
import tf2_geometry_msgs
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from slam_toolbox.srv import SaveMap
from std_msgs.msg import String

from priority_queue import PriorityQueue 

#TODO: 
#fix dummy planner client line line 381
# fix risk map generation to only generate x number of layers. 
# stop earlier when there is no novelty over x value
# improve goal finding to remember prev goal and not allow goals within that range
# ensure proper gain values for A*
# Make the goal be constantly evaluated to see if it is in free cells. do last (if this problem keeps appearing)

# feedback / replan thresholds
REPLAN_THRESHOLD = 0.5
REPLAN_CUTOFF    = 0.8

# Add alongside your other constants at the top of dummy_planner_client.py
FRONTIER_BLOB_WEIGHT = 1.0   # reward per frontier cell in the blob
FRONTIER_DIST_WEIGHT = 2.0   # penalty per metre to the blob centroid
MIN_BLOB_AREA        = 5     # blobs smaller than this are ignored (noise)

class DummyClient(Node):

    def __init__(self):
        super().__init__('Dummy_Path_Planner_Client')
        self.cb_group = ReentrantCallbackGroup() 
        
        #Action client to controller
        self._action_client = ActionClient(self, PathGen, "pathgen")

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability= ReliabilityPolicy.RELIABLE,
            durability = DurabilityPolicy.TRANSIENT_LOCAL,
            )
        #Service Client to path planner
        self.cli = self.create_client(GetPlan, 'request_plan',callback_group=self.cb_group)

        #publishers:
        self.publisher_ = self.create_publisher(Path, '/nav_path', 10)

        #subscribters:
        self.create_subscription(OccupancyGrid, '/frontier', self.handle_frontier, 10)
        self.create_subscription(Odometry, '/odom', self.update_odom, 10)
        self.create_subscription(OccupancyGrid, '/map/safe', self.handle_map, map_qos)
        self.create_subscription(PointStamped, '/clicked_point', self.handle_clicked_point, 10)

        self.create_subscription(OccupancyGrid, '/risk', self.handle_risk_map, 10)
        self.create_subscription(OccupancyGrid, '/novelty', self.handle_novelty_map, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.handle_goal, 10)
        
        # #request a path
        self.req = GetPlan.Request()

        # Add these to track the current target and the replan loop
        self.current_goal_msg   = None
        self.replan_timer       = None
        self.is_exploring       = True
        self.goal_handle        = None
        self.frontier_map       = None 
        self.done               = False
        self.best_centroid      = None
        self.safe_map           = None
        self.map_info           = None
        self.novelty_map        = None
        self.risk_map           = None
        self.odom               = Odometry()
        self.ordered_goals      = PriorityQueue()
        self.phase              = 1
        self.start_pointStamped = PointStamped()
        self.is_replanning      = True
        self.start_time         = self.get_clock().now()

        #timer for ending early, if in a log path
        # self.timer = self.create_timer(1.0, self.check_done)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


# ------------------------------------------------------------------------------------
# Path Planner Client Bridge
#-------------------------------------------------------------------------------------

    def send_request(self, goal:PoseStamped):
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('request_plan service not available, dropping request.')
            return
        goal_msg = self.tf_buffer.transform(goal, "map", rclpy.duration.Duration(seconds=1))
        #get robots pose in map frame
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'base_link'
        # start_pose.header.stamp = goal.header.stamp
        
        start_pose = self.tf_buffer.transform(start_pose, 'map', rclpy.duration.Duration(seconds=1))
        start_pose.pose.position.z = 0.0


        self.req.start = start_pose
        
        self.req.goal = goal_msg


        self.future = self.cli.call_async(self.req)

        self.future.add_done_callback(self.handle_response)


    def handle_response(self,future):
        try:
            result = future.result()
            if result is None or not result.plan.poses:
                self.get_logger().warn(f'Path too short/DNE ({len(result.plan.poses) if result else 0} poses). Replanning...')
                
                if self.ordered_goals.empty(): #No goals left 
                    self.get_logger().info('Exploration Complete: No more goals in queue.')
                    if self.phase != 0:
                        grid_start = self.world_to_grid(self.map_info,self.start_pointStamped.point)
                        self.find_and_send_path(grid_start)
                    else:
                        self.get_logger().error(' no path valid to the start :()')
                    self.phase = 0 # Idle phase
                    return
        
                else: #the goal it gave was garbage or unreachable replan
                    new_goal = self.ordered_goals.get()
                    self.find_and_send_path(new_goal)
                    return 
                
            goal_msg = PathGen.Goal()
            goal_msg.path = result.plan 

            self._action_client.wait_for_server()
            self._send_goal_future = self._action_client.send_goal_async(
                goal_msg, feedback_callback=self.feedback_callback
            )
            self._send_goal_future.add_done_callback(self.goal_response_callback)
        
            
            self.publisher_.publish(result.plan)
        except Exception as e:
            self.get_logger().error(f'Service call failed in handle response Phase 1: {e}')
            return 

# ------------------------------------------------------------------------------------
# Action Client Bridge
#-------------------------------------------------------------------------------------
    
    def send_path(self, path):
        """Send a list of PoseStamped poses to the controller action server."""
        if not path or len(path) <= 1:
            self.get_logger().warn('path is not of sufficient length skipping.')
            return
            
        path_obj = Path()
        path_obj.poses = path
        path_obj.header.frame_id = 'map'
        path_obj.header.stamp = self.get_clock().now().to_msg()

        goal_msg = PathGen.Goal()
        goal_msg.path = path_obj
        self.done = False

        self._action_client.wait_for_server(timeout_sec=1)
        self.publisher_.publish(path_obj)
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        return self._send_goal_future
    

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self.done = False
        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}')
        
        if self.is_replanning == True:
            if not result.success:
                self.get_logger().warn('Path failed, skipping replan.')
                return
            
            self.find_and_send_path() #done pp replan!
            return
        
        else:
            self.save_map_and_finish()
            return

        
    def feedback_callback(self, feedback_msg):
        replan = False #debug flag that ensure if there needs to be a new goal a new path will be created. 
        if self.done:
            return

        feedback = feedback_msg.feedback.percent_complete
        return 
        # if feedback >= REPLAN_CUTOFF: #defined at top of page
        #     return
        
        # if feedback > REPLAN_THRESHOLD * self.replan_counter: #defined at top of page
        #     self.replan_counter += 1
        #     self.get_logger().info(f'Replanning at {feedback*100:.1f}% completion...')
        #     self.find_and_send_path(goal=self.goal)

            

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceling...')
        else:
            self.get_logger().info('Goal failed to cancel.')

    def request_cancel(self):
        if self.goal_handle is not None:
            self.get_logger().info('Requesting cancel...')
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(self.cancel_done_callback)

#------------------------------------------------------------------------------
# Robot pose helper function
# ----------------------------------------------------------------------------- 
    def update_odom(self, msg: Odometry):
        self.odom = msg
    
    def get_robot_pose(self):
        try:            
            transform = self.tf_buffer.lookup_transform(
                'map', self.odom.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1)
            )
            pose_stamped = PoseStamped()
            pose_stamped.header = self.odom.header
            pose_stamped.pose   = self.odom.pose.pose   # unwrap PoseWithCovariance → Pose
            msg_transformed = tf2_geometry_msgs.do_transform_pose_stamped(pose_stamped, transform)
           
            px = msg_transformed.pose.position.x
            py = msg_transformed.pose.position.y
            quat = msg_transformed.pose.orientation
            (roll, pitch, yaw) = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler("xyz")
            self.ptheta = yaw 
            return (px,py,yaw)

        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None

    
    def is_path_significantly_different(self, coords_a, coords_b, threshold=0.1):
        """
        Compares two paths to see if the deviation exceeds a threshold (meters).
        :param path_a: List of (x,y) tuples (or list of PoseStamped)
        :param path_b: List of (x,y) tuples (or list of PoseStamped)
        :param threshold: Distance in meters to consider 'significant'
        """        

        if not coords_a or not coords_b:
            return True # If one is empty, it's definitely different

        # 2. Resample or align lengths (Simplified: Check points along the shorter length)        
        total_deviation = 0.0
        check_len = min(len(coords_a),len(coords_b))
        for i in range(check_len):
            dist = math.dist(coords_a[i], coords_b[i])
            total_deviation += dist

        avg_deviation = total_deviation / check_len

        # 3. Logic: If the average deviation is > 15cm, it's a new path
        return avg_deviation > threshold
    
#------------------------------------------------------------------------------
# CALLBACK FOR PLOTTING POINT
# ----------------------------------------------------------------------------- 
    def handle_clicked_point(self, msg:PointStamped):
        # self.save_map_and_finish("rbe3002_test_map")
        self.phase = 1 
        world_start = self.get_robot_pose()[:2]
        self.start_pointStamped.point = Point(x=world_start[0],y=world_start[1],z=0.0)
        self.start_pointStamped.header.stamp = rclpy.time.Time().to_msg()
        self.start_pointStamped.header.frame_id = 'map'
        self.start_time         = self.get_clock().now()

        self.find_and_send_path()

    def handle_goal(self, msg: PoseStamped):
        self.is_replanning = False
        self.is_exploring = False
        world_start = self.get_robot_pose()[:2]
        self.start_pointStamped.point = Point(x=world_start[0],y=world_start[1],z=0.0)
        self.start_pointStamped.header.stamp = rclpy.time.Time().to_msg()
        self.start_pointStamped.header.frame_id = 'map'

        self.phase = 2
        self.get_logger().info(f'going to point!')
        self.send_request(goal=msg)


    def find_and_send_path(self, goal=[-1, -1]):
        self.replan_counter = 1         
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1)
            )
        except Exception as e:
            self.get_logger().error(f'Could not get robot pose: {e}')
            raise Exception('No potential goals — TF unavailable')

        start_point = PointStamped()
        start_point.header.frame_id = 'base_link'
        start_point.header.stamp = rclpy.time.Time().to_msg()
        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        goal_pose = PoseStamped()

        goal_pose.header.frame_id = 'map'          
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        if goal == [-1, -1]:
            goalx, goaly = self.find_reasonable_goal()
        else:
            goalx, goaly = goal

        self.goal = (goalx,goaly)
        goal_point = self.grid_to_world(self.map_info,(goalx,goaly))
        goal_pose.pose.position.x = goal_point.x
        goal_pose.pose.position.y = goal_point.y
        goal_pose.pose.position.z = 0.0
        

        self.send_request(goal_pose)


    def handle_map(self,map:OccupancyGrid):
        safe_map = np.asarray(map.data, dtype=np.int32).reshape(map.info.height, map.info.width)
        self.safe_map = safe_map
        self.map_info = map.info


    def handle_risk_map(self, map:OccupancyGrid):
        if self.map_info != None:
            if map.info.height == self.map_info.height and map.info.width == self.map_info.width:
                risk_map = np.asarray(map.data, dtype=np.int32).reshape(map.info.height, map.info.width)
                self.risk_map = risk_map
                # self.get_logger().info("risk found")
            else:
                self.get_logger().info("maps wrong size")


    def handle_novelty_map(self, map:OccupancyGrid):
        if self.map_info != None:
            if map.info.height == self.map_info.height and map.info.width == self.map_info.width:
                novelty_map = np.asarray(map.data, dtype=np.int32).reshape(map.info.height, map.info.width)
                self.novelty_map = novelty_map
                # self.get_logger().info("novelty found")
            else:
                self.get_logger().info("maps wrong size")

    
    def handle_frontier(self, frontier_map: OccupancyGrid):
        """
        Groups frontier cells (8-connected) into blobs via OpenCV CCL, scores each as:
            score = (FRONTIER_BLOB_WEIGHT * num_cells) - (FRONTIER_DIST_WEIGHT * distance_m)
        Stores and returns the world-space (wx, wy) of the highest scoring blob.
        """
        self.frontier_map = frontier_map

    


    def find_reasonable_goal(self):

        if self.safe_map is None or self.novelty_map is None or self.risk_map is None:
            self.get_logger().info("maps missing")
            return self.world_to_grid(self.map_info, self.start_pointStamped.point)
        
        if self.safe_map.shape != self.risk_map.shape or self.novelty_map.shape != self.risk_map.shape or self.safe_map.shape != self.novelty_map.shape:
            self.get_logger().warn('maps of different sizes, trying to find a new goal')
            return self.find_reasonable_goal()


        safe_mask = np.uint8(self.safe_map.astype(np.uint8) == 0)

        cost_map = np.zeros(self.safe_map.shape)

        safe_risk = np.clip(self.risk_map, 1, 128)
        cost_map = self.novelty_map * (128/safe_risk) * safe_mask

        blurred = gaussian_filter(cost_map.astype(np.float64), sigma=0.5)

        # Build 8-neighbor comparison using slices — no loops needed.
        # Interior region (avoids border pixels, which lack a full neighborhood).
        center = blurred[1:-1, 1:-1]

        neighbors = [
            blurred[0:-2, 0:-2],  # top-left
            blurred[0:-2, 1:-1],  # top
            blurred[0:-2, 2:  ],  # top-right
            blurred[1:-1, 0:-2],  # left
            blurred[1:-1, 2:  ],  # right
            blurred[2:,   0:-2],  # bottom-left
            blurred[2:,   1:-1],  # bottom
            blurred[2:,   2:  ],  # bottom-right
        ]

        # A pixel is a local max if it's strictly greater than every neighbor.
        is_max = np.ones(center.shape, dtype=bool)
        for neighbor in neighbors:
            is_max &= center > neighbor

        # Convert boolean mask back to full-image coordinates (+1 offset for border).
        rows, cols = np.where(is_max)
        
        best_points = list(zip(cols + 1, rows + 1))

        robot_pose = self.get_robot_pose()
        robot_point = Point(x=robot_pose[0], y=robot_pose[1], z =0.0)
        robot_grid_pose = self.world_to_grid(self.map_info, robot_point)
        
        ordered_goals = PriorityQueue()

        for gx, gy in best_points:
            value = cost_map[gy, gx]
            distance = math.dist(robot_grid_pose, (gx, gy))

            dx = gx - robot_grid_pose[0]
            dy = gy - robot_grid_pose[1]
            path_heading = math.atan2(dy, dx)

            diff = abs((robot_pose[2] - path_heading + math.pi) % (2 * math.pi) - math.pi)
            heading_cost = diff / math.pi

            VALUE_WEIGHT = 5
            DISTANCE_WEIGHT = 4   
            HEADING_WEIGHT = 70 #radians

            priority = -1 * ((value*VALUE_WEIGHT) - (distance*DISTANCE_WEIGHT) - (heading_cost*HEADING_WEIGHT))

            if distance < 3:
                priority = 1000

            if value > 20:
                ordered_goals.put((gx, gy), priority)



            # self.get_logger().info(f'goal position: {gx} {gy}')


        self.ordered_goals = ordered_goals
        self.get_logger().info(f'number of goals considered {len(ordered_goals.get_queue())}')

        if ordered_goals.empty():
            self.get_logger().info('going home')
            self.is_replanning = False
            return self.world_to_grid(self.map_info, self.start_pointStamped.point)
        else: 
            best_value = ordered_goals.get()


        self.get_logger().info(f'best value: {best_value[0]}, {best_value[1]}')

        return best_value


    # def check_done(self):
    #     # pass
    #     goal = self.find_reasonable_goal()
    #     if goal == self.world_to_grid(self.map_info, self.start_pointStamped.point) and self.goal:
    #         s
    #         self.request_cancel()
    #         self.find_and_send_path(goal)
#------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------- 
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
    
    #-------------------------------------------------------------------------------
    # Saving the Map
    #-------------------------------------------------------------------------------

    # Inside your Dummy Client class:
    def save_map_and_finish(self, filename="final_lab_map"):
        if self.is_exploring:
            curr_time = self.get_clock().now()
            total_time = curr_time - self.start_time 
            seconds = total_time.nanoseconds / 1e9
            self.get_logger().info(f'Mission Complete! time = {seconds} Attempting to save map...')
            
            # Create the client for slam_toolbox
            save_client = self.create_client(SaveMap, '/slam_toolbox/save_map')
            
            while not save_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for SLAM Toolbox save service...')

            # Prepare the request
            req = SaveMap.Request()
            req.name.data = filename + str(seconds) # slam_toolbox expects a String message type here
            
            # Call the service
            future = save_client.call_async(req)
            future.add_done_callback(self.save_response_callback)

    def save_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info('Map saved successfully! You can now Ctrl+C.')
        except Exception as e:
            self.get_logger().error(f'Service call failed in saving map: {e}')

def main(args=None):
    rclpy.init(args=args)

    node = DummyClient()
    executor = MultiThreadedExecutor()

    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()