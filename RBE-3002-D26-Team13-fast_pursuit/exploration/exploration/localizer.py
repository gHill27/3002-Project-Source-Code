#!/usr/bin/env python3

import rclpy
import math
import numpy as np
import tf2_geometry_msgs
import time

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse

from custom_action_interfaces.action import PathGen
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped, Twist, Pose, PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from typing import Optional
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from math import atan2, sqrt


class Localizer(Node):

    def __init__(self):
        # Initialize node, name it 'localizer'
        super().__init__("localizer")
        self.get_logger().info('localizer started')
        
        # Create callback groups
        self.cb_group = ReentrantCallbackGroup()
        self.cb_driving = MutuallyExclusiveCallbackGroup()
        
        # Subscribe to necessary topics
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.update_covariance, 10, callback_group=self.cb_group
        )

        # Publish velocities to cmd_vel
        self.vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        # Initalize amcl covariance
        self.x_cov = 100.0
        self.y_cov = 100.0
        self.yaw_cov = 100.0


        # Create a timer and call check callback
        self.timer = self.create_timer(1, self.timer_callback)


    def update_covariance(self, msg: PoseWithCovarianceStamped):
        
        covariance = msg.pose.covariance

        self.x_cov = covariance[0]
        self.y_cov = covariance[7]
        self.yaw_cov = covariance[35]

        if self.x_cov > 0.01 and self.y_cov > 0.01 and self.yaw_cov > 0.01 and self.timer.is_canceled:
            self.timer.reset()



    def timer_callback(self):

        if self.x_cov < 0.01 and self.y_cov < 0.01 and self.yaw_cov < 0.01:

            self.send_speed(0.0, 0.0)

            self.get_logger().info("*******************************")
            self.get_logger().info("Localized, ready for goal poses")
            self.get_logger().info("*******************************")
            
            time.sleep(1)
            
            self.timer.cancel()


        else:
            
            self.send_speed(0.0, 0.5)


    def send_speed(self, linear_speed: float, angular_speed: float):
        """
        Sends speed to the /cmd_vel topic, which runs the turtlebot motors.
        :param linear_speed  [float] [m/s]   The forward linear speed.
        :param angular_speed [float] [rad/s] The angular speed for rotating around the body center.
        """

        # Create TwistStamped object with given linear and angular speeds
        msg_cmd_vel = TwistStamped()

        msg_cmd_vel.twist.linear.x = linear_speed    # linear speed is always in x
        msg_cmd_vel.twist.angular.z = angular_speed  # angular speed is always in y

        # Publish TwistStamped to cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)
    

def main(args=None):
    rclpy.init(args=args)

    localizer = Localizer()
    executor = MultiThreadedExecutor()
    executor.add_node(localizer)

    try:
        executor.spin()
    except KeyboardInterrupt:  # when 'ctrl + C' is pressed
        pass
    finally:
        localizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()