#!/usr/bin/env python3

import rclpy
import math
import numpy as np
import tf2_geometry_msgs
import time

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.action import GoalResponse 

from custom_action_interfaces.action import PathGen
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped, Twist, Pose
from scipy.spatial.transform import Rotation
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from typing import Optional
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from math import atan2, sqrt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
DIST_AHEAD   = 0.2     # pure-pursuit lookahead distance (m)
LIN_SPEED    = 0.22     # maximum linear speed (m/s)
ANG_SPEED    = 2.8        # max ang speed
# LIN_SPEED    = 0.1
# ANG_SPEED    = 1.0
PP_TOLERANCE = 0.04     # goal-reached radius (m)
CTRL_RATE    = 10       # control loop rate (Hz)
KP           = 3        # angular proportional gain
 

class Controller(Node):
    def __init__(self):
        # Initialize node, name it 'controller'
        super().__init__("controller")
        self.get_logger().info('controller started')

        # Create callback groups
        self.started = False
        self.cb_group = ReentrantCallbackGroup()
        self.cb_driving = MutuallyExclusiveCallbackGroup()

        
        self._action_server = ActionServer(
            self, PathGen, "pathgen", 
            self.execute_callback,
            goal_callback=self.goal_callback, 
            cancel_callback=self.cancel_callback, 
            callback_group=self.cb_group
        )
        

        # Subscribe to necessary topics
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.update_odometry, 10, callback_group=self.cb_group
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/move_base_simple/goal",
            self.go_to,
            10,
            callback_group=self.cb_driving
        )

        # Publish 
        self.vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.local_goal_marker_pub = self.create_publisher(
            Marker, '/pure_pursuit/local_goal_marker', 10
        )

        # Initalize robot pose (x, y, theta)
        self.px = 0.0
        self.py = 0.0
        self.ptheta = 0.0

        self.perc_path_complete = 0.0
        self.active_goal_handle = None # Track the current running goal

        self.started = False
        # Create a transform listener + TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------
    def execute_callback(self, goal_handle):
        if self.active_goal_handle is not None and self.active_goal_handle.is_active:
            self.get_logger().info('Preempting active goal...')
            try:
                self.active_goal_handle.abort() 
            except Exception as e:
                self.get_logger().warn(f"Could not abort previous goal: {e}")

        self.active_goal_handle = goal_handle
        self.get_logger().info('Executing goal...')
        
        path = goal_handle.request.path

        self.handle_path_pp(path, goal_handle)
        # self.handle_path(path)
        result = PathGen.Result()
        if len(path.poses) <= 1:
            self.get_logger().warn('Path too short, succeeding immediately.')
            result.success = True
            if goal_handle.is_active:          
                goal_handle.succeed()
            self.started = False
            return result

        if not goal_handle.is_active:
            self.get_logger().info('Goal state is CANCELED. Returning result without succeeding.')
            result.success = False
            self.started = False
            return result
        
        try:
            goal_handle.succeed()
            result.success = True

        except Exception as e:
            self.get_logger().warn(f'could not succeed goal handle: {e}')
            result.success = False
    
        # If we got here but the handle isn't active, it was replaced or canceled
        self.started = False
        return result
    
    def goal_callback(self, goal_request):
        """Accepts the new goal and aborts the old one if it exists."""
        self.get_logger().info('Received new path request - preempting old goal')
        
        if self.active_goal_handle is not None and self.active_goal_handle.is_active:
            self.get_logger().info('Aborting previous goal to follow new plan.')
        
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        self.started = False
        return CancelResponse.ACCEPT
     
    #--------------------------------------------------------------------
    # odom
    #=-------------------------------------------------------------------
     
    def update_odometry(self, msg: Odometry):
        """
        A callback that updates the current pose of the robot
        :param msg [Odometry] the current odometry information
        """
        try:
            # Extract pose from odometry and put it in a PoseStamped
            
            transform = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1)
            )

            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = msg.pose.pose   
            msg_transformed = tf2_geometry_msgs.do_transform_pose_stamped(pose_stamped, transform)
            
            self.px = msg_transformed.pose.position.x
            self.py = msg_transformed.pose.position.y
            quat = msg_transformed.pose.orientation
            (roll, pitch, yaw) = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler("xyz")
            self.ptheta = yaw 

        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None

    #--------------------------------------------------------------------
    # velocity
    #=-------------------------------------------------------------------
    def send_speed(self, linear_speed: float, angular_speed: float):
        """
        Sends speed to the /cmd_vel topic, which runs the turtlebot motors.
        :param linear_speed  [float] [m/s]   The forward linear speed.
        :param angular_speed [float] [rad/s] The angular speed for rotating around the body center.
        """

        # Create TwistStamped object with given linear and angular speeds
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.header.stamp = self.get_clock().now().to_msg()

        msg_cmd_vel.twist.linear.x = float(linear_speed)    # linear speed is always in x
        msg_cmd_vel.twist.angular.z = float(angular_speed)  # angular speed is always in y

        # Publish TwistStamped to cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)

    #--------------------------------------------------------------------
    # Pure Pursuit
    #=-------------------------------------------------------------------
    def handle_path_pp(self, path: Path, goal_handle):
        """
        A callback function that handles iterating through a path using pure pursuit.
        ASSUME ROBOT STARTS ON PATH
        :param path [Path] The path that the robot will pursue.
        """
        
        if not path.poses:
            self.get_logger().error("PLANNER ERROR: Received an empty path! The goal is likely unreachable.")
            self.send_speed(0.0, 0.0)
            
            # Create a result object to return to the planner
            result = PathGen.Result()
            result.success = False
            
            goal_handle.abort() 
            return # Exit the function immediately
        
        path_coords = []
        for element in path.poses:
            goal_msg = self.tf_buffer.transform(
            element, "map", rclpy.duration.Duration(seconds=1)
            )
            path_coords.append((goal_msg.pose.position.x, goal_msg.pose.position.y))

        path_coords = self.clean_path(path_coords)

        total_len = self.find_total_path_len(path_coords)

        if total_len == 0.0:
            self.get_logger().error("Zero-length path, aborting.")
            goal_handle.abort()
            return
    
        current_line_index = self.find_nearest_segment(path_coords,self.px,self.py) 
        previous_point_on_line = path_coords[current_line_index- 1]
        line_end = path_coords[-1]
        rate = self.create_rate(CTRL_RATE)

        self.perc_path_complete = 0.0
        
        # Find point on line after projecting a distance
        local_goal = self.find_local_goal(path_coords, previous_point_on_line, current_line_index, DIST_AHEAD)

        while rclpy.ok():
            # Check if THIS specific goal is still the one we should be following
            if not goal_handle.is_active:
                self.get_logger().info("Current path goal no longer active (Preempted). Exiting loop.")
                return
            

            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal canceled!')
                self.send_speed(0.0, 0.0) # STOP THE ROBOT
                goal_handle.canceled()    # Update the action state
                return # Exit the function
            

            # Update necessary variables for next goal
            smallest_dist = math.inf
            closest_point = (self.px, self.py)
            # current_line_index = max(current_line_index, local_goal[1])  # advance to goal's segment
            search_start = max(1, current_line_index-1)
            search_end = local_goal[1] + 1
            local_goal = self.find_local_goal(path_coords, closest_point, current_line_index, DIST_AHEAD)
            
            
            # Loop through path segments from current one up to the segment where the goal is,
            # find the distance from the robot to each line to find the closest point
            for i in range(search_start,search_end):
                mutated_path = [path_coords[i - 1], path_coords[i]]
                point = (self.px, self.py)
                location, _ , distance_to_line = self.find_closest_point_on_line(point, mutated_path)

                # Compare found distance to running variable to track the smallest one,
                if distance_to_line < smallest_dist:
                    smallest_dist = distance_to_line
                    closest_point = location
                    current_line_index = max(current_line_index, i)

            local_goal_data = self.find_local_goal(path_coords, closest_point, current_line_index, DIST_AHEAD)
            target_pt = local_goal_data
            dist_traveled = self.find_dist_to_point(
                path_coords, local_goal[1], local_goal[0]
            )
            self.perc_path_complete = min(dist_traveled / total_len, 1.0)
        
            # Find and normalize the angle to the goal position
            y_diff = target_pt[0][1] - self.py
            x_diff = target_pt[0][0] - self.px

            angle = math.atan2(y_diff, x_diff) 
            error = angle - self.ptheta
            error = self.normalize_angle(error)
            lin_speed = max(LIN_SPEED * (1 - abs(error/(math.pi))), 0.01) #TODO fix later!!!!!
            if error > (math.pi/2):
                lin_speed = 0.0
            # Calculate the angular speed and publish it along with linear speed
            ang_speed = max(-ANG_SPEED, min((KP * error), ANG_SPEED))
            self.send_speed(lin_speed, ang_speed)


            self.publish_local_goal_marker(local_goal[0][0],local_goal[0][1])
            # If the robot has reached the end of the path, stop
            if math.dist((self.px, self.py), line_end) < PP_TOLERANCE:
                self.send_speed(0.0,0.0)
                break

            feedback_msg = PathGen.Feedback()
            feedback_msg.percent_complete = self.perc_path_complete
            # self.get_logger().info(f'Feedback: {self.perc_path_complete}')

            goal_handle.publish_feedback(feedback_msg)
            rate.sleep()

    # ------------------------------------------------------------------
    # Pure-pursuit geometry helpers
    # ------------------------------------------------------------------

    def find_total_path_len(self,path):
        distance = 0.0
        for index in range(len(path)-1):
            distance += math.dist(path[index],path[index+1])
        return distance


    def find_closest_point_on_line(self, robot_coords, mutated_path):
        """
        Finds the closest point on a path line from a robot in space
        :param robot_coords [tuple (x,y)]  Robot position
        :param mutated_path [list] Line segment from a path line
        """
        line_start = np.asarray(mutated_path[0], dtype=float)
        line_end = np.asarray(mutated_path[1], dtype=float)
        robot_pos = np.asarray(robot_coords, dtype=float)

        line_dir = line_end - line_start  # direction vector derived from the two points

        #Make sure the points aren't on each other (i.e. actually a line)
        if np.allclose(line_dir, 0):
            self.get_logger().error(f'line_end: {line_end}, \n line_start {line_start}')
            raise ValueError("line_p1 and line_p2 must be distinct points")
            # return tuple(line_start), 0.0, 0.0

        # Find how far along the segment the closest point to the robot is,
        # constrain is from [0, 1] so it stays on the actual line
        dist_along = np.dot(robot_pos - line_start, line_dir) / np.dot(line_dir, line_dir)
        dist_along = np.clip(dist_along, 0.0, 1.0)

        # Find coordinates of the closest point along the line, the distance from
        # the robot to that point, and cast the coordinates to a tuple for outside use
        closest_coords = line_start + dist_along * line_dir
        distance = np.linalg.norm(robot_pos - closest_coords)
        closest_coords = tuple(closest_coords)

        return (float(closest_coords[0]),float(closest_coords[1])), float(dist_along), float(distance) 

    def clean_path(self, path_coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Removes consecutive duplicate points from the path.
        """
        if not path_coords:
            return path_coords
        
        cleaned = [path_coords[0]]
        for point in path_coords[1:]:
            if point != cleaned[-1]:
                cleaned.append(point)
            else:
                self.get_logger().warn(f"Removed duplicate point: {point}")
        
        return cleaned

    def find_local_goal(
        self, path: list[tuple[float, float]], point, index, distance_ahead: float
    ) -> tuple[tuple[float, float], int]:
        """
        Finds the next local goal point a given distance ahead along the path.
        :param path  [list[tuple[float, float]]] The list of path coordinates.
        :param point  [tuple (x,y)] The current closest point on the path.
        :param index  [int]  The current line segment index.
        :param distance_ahead [float]  The lookahead distance along the path.
        """
        remaining = distance_ahead
        distance_traveled = 0.0
        working_path = list(path)
        if index > 0:
            working_path[index - 1] = point # Edit path so first line segment starts from point

        # Loop through path and check if goal is in current line segment
        for i in range(index - 1 , len(working_path)-1):
            line_length = math.dist(working_path[i], working_path[i + 1])

            # If the goal is on the current segment, find the goal coordinates
            if line_length >= remaining:
                angle = math.atan2(
                    working_path[i + 1][1] - working_path[i][1],
                    working_path[i + 1][0] - working_path[i][0],
                )
                goalx = remaining*math.cos(angle) + working_path[i][0]
                goaly = remaining*math.sin(angle) + working_path[i][1]
                return ((goalx, goaly), i+1) 

            # If the goal is ahead of the current segment, reduce the distance
            # for comparing to the next segment
            else:
                remaining -= line_length

        # At the end of the path, target the last point
        return ((working_path[-1][0], working_path[-1][1]), len(working_path)-1)

    def find_total_path_len(self, path):
        distance = 0.0
        for i in range(len(path) - 1):
            distance += math.dist(path[i], path[i + 1])
        return distance
    
    def find_dist_to_point(self, path, index, point):
        """
        Returns cumulative arc-length from path[0] to `point`,
        where point lies on segment (index-1 → index).
        """
        distance = 0.0
        for i in range(1, index):
            distance += math.dist(path[i - 1], path[i])
        # add the partial segment from path[index-1] to the projected point
        distance += math.dist(path[index - 1], point)
        return distance
    
    def find_nearest_segment(self, path_coords: list, robot_x: float, robot_y: float) -> int:
        """
        Returns the index i (1-based) of the segment path_coords[i-1]→path_coords[i]
        that is closest to the robot's current position.
        """
        best_index = 1
        best_dist = math.inf
        for i in range(1, len(path_coords)):
            segment = [path_coords[i - 1], path_coords[i]]
            _, _, dist = self.find_closest_point_on_line((robot_x, robot_y), segment)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return best_index


# ------------------------------------------------------------------
# Primitive motion commands
# ------------------------------------------------------------------

    def drive(self, distance: float, linear_speed: float):
        """
        Drives the robot in a straight line.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The forward linear speed.
        """
        # Define constants amd rate object
        RATE_REFRESH = 10
        TOLERANCE = 0.02
        rate = self.create_rate(RATE_REFRESH)

        try:
            # Store initial coordinates
            init_x = self.px
            init_y = self.py

            while rclpy.ok():
                # Loop unti distance traveled is within tolerance of target
                curr_dist = math.dist((init_x, init_y), (self.px, self.py))
                if(abs(curr_dist - distance) >= TOLERANCE):
                    break

                # Continuously send speed and sleep
                self.send_speed(linear_speed, 0.0)
                rate.sleep()

            # When distance is reached, stop
            self.send_speed(0.0, 0.0)

        except TypeError:
            self.get_logger().warn("No Robot Pose for Driving")


    def rotate(self, angle: float, angular_speed: float):
        """
        Rotates the robot around the body center by the given angle.
        :param angle         [float] [rad]   The distance to cover.
        :param angular_speed [float] [rad/s] The angular speed.
        """
        # Define constants amd rate object
        RATE_REFRESH = 10
        TOLERANCE = 0.1
        rate = self.create_rate(RATE_REFRESH)

        try:
            # Store initial robot angle
            init_theta = self.ptheta
            angle = self.normalize_angle(angle)

            direction = 1.0 if angle >= 0 else -1.0
            
            while rclpy.ok():
                # Normalize angle and determine shortest direction
                curr_angle = self.normalize_angle(self.ptheta - init_theta)

                # Loop until angular distance travelled is within tolerance
                error = curr_angle - angle
    
                if abs(error) <= TOLERANCE:
                    break
    
                # Continuously send speed and sleep
                self.send_speed(0.0, direction * angular_speed)
                rate.sleep()
    
            # When turn distance is reached, sleep
            self.send_speed(0.0, 0.0)

        except TypeError:
            self.get_logger().warn("No Robot Pose For Turning")


    def go_to(self, msg: PoseStamped):
        """
        Uses rotate() and drive() to get to a specific pose.
        :param msg [PoseStamped] The target or "goal" pose.
        """
        self.get_logger().info('going to a new point')
        # Translate PoseStamped message to map frame
        pose_msg = self.tf_buffer.transform(msg, "map", rclpy.duration.Duration(seconds=1))
        
        # Get target pose variables
        goal_x = pose_msg.pose.position.x
        goal_y = pose_msg.pose.position.y
        quat = pose_msg.pose.orientation

        (roll, pitch, yaw) = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler("xyz")
        goal_angle = yaw

        # Calculate first turning angle before driving and rotate
        drive_angle = atan2(goal_y - self.py, goal_x - self.px) - self.ptheta
        self.rotate(drive_angle, 1.0)

        # Calculate distance to goal point and drive distance
        target_distance = sqrt((goal_x - self.px)**2 + (goal_y - self.py)**2)
        self.smooth_drive(target_distance, 0.5)

        # Calculate final target angle and rotate
        target_angle = goal_angle - self.ptheta
        self.rotate(target_angle, 1.0)


    def handle_path(self, path: Path):
        """
        A callback function that handles iterating through a path.
        :param path [Path] The path that the robot will drive.
        """
        self.get_logger().info("Started traversing the path!")
        for step in path.poses:
            self.go_to(step)
    

    def smooth_drive(self, distance: float, linear_speed: float):
        """
        Smoothly drives the robot in a straight line by regulating its speed.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The maximum forward linear speed.
        """
        # Create constants and variables
        MAX_ACCELERATION = 0.2
        TOLERANCE = 0.01
        rate = self.create_rate(10)
        init_x = self.px
        init_y = self.py
        counter = 0

        while True:
            # Continuously update distance traveled relative to initial x and y
            distance_traveled = math.dist(
                (init_x, init_y),
                (self.px, self.py),
            )

            # Stop when the robot has reached the target within a tolerance
            distance_to_travel = abs(distance_traveled - distance)
            if (distance_to_travel <= TOLERANCE):
                break

            # logic for controlling smooth speed
            # -----------------------------------
            counter += 1
            # uses 10hz so we multiply by 1/10 to convert.
            currTime = counter * (1 / 10)
            # determines current speed through our allowed maximum acceleration
            target_speed = currTime * MAX_ACCELERATION
            # bound the speed to the speed given above. (should be a max of ~ 0.5m/s )
            if target_speed > linear_speed:
                target_speed = linear_speed

            # logic for stopping
            # -----------------------------------
            # kinematic equation #3
            stop_distance = (target_speed**2) / (2 * MAX_ACCELERATION)
            if distance_to_travel <= stop_distance:
                target_speed = math.sqrt(2 * MAX_ACCELERATION * distance_to_travel)
            # sends the target speed with no angular velocity
            self.send_speed(target_speed, 0.0)
            rate.sleep()  # sleep for threading

        self.send_speed(0.0, 0.0)  # clean stop when done driving.

    def normalize_angle(self, angle: float) -> float:
        """
        Wraps an angle to the range (-π, π] to avoid discontinuity issues
        near ±180°
        :param angle [float] [rad] The raw angle.
        :return      [float] [rad] The equivalent angle in (-π, π].
        """
        while angle >  math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
# ------------------------------------------------------------------
# RViz marker
# ------------------------------------------------------------------
    def publish_local_goal_marker(self, x: float, y: float):
        """
        Publishes a sphere Marker to /pure_pursuit/local_goal_marker so that
        the current pure-pursuit lookahead goal is visible in Rviz.
        :param x [float] X position in the map frame.
        :param y [float] Y position in the map frame.
        """
        marker = Marker()

        # Frame and namespace
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit"
        marker.id = 0                          # single marker, always overwrite
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position — keep z=0 so it sits on the ground plane
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0        # identity quaternion

        # Size: diameter equal to the lookahead distance so it's easy to gauge
        MARKER_DIAMETER = 0.08                 # metres — tweak to taste
        marker.scale.x = MARKER_DIAMETER
        marker.scale.y = MARKER_DIAMETER
        marker.scale.z = MARKER_DIAMETER

        # Bright cyan so it stands out against the costmap
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0                   # fully opaque

        # Keep the marker alive for 0.5 s so it disappears if pursuit stops
        marker.lifetime = Duration(sec=0, nanosec=500_000_000)

        self.local_goal_marker_pub.publish(marker)

    


def main(args=None):
    rclpy.init(args=args)

    controller = Controller()
    executor = MultiThreadedExecutor()
    executor.add_node(controller)

    try:
        executor.spin()
    except KeyboardInterrupt:  # when 'ctrl + C' is pressed
        pass
    finally:
        executor.shutdown()
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()