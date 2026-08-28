#!/usr/bin/env python3

import rclpy

from typing import Tuple, List
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from custom_action_interfaces.action import PathGen

class PathGenerator(Node):
    def __init__(self, px, py, pth, frame):
        super().__init__('path_generator')

        self._action_client = ActionClient(self, PathGen, "pathgen")

        self.px = px
        self.py = py
        self.pth = pth
        self.frame = frame

        # note the topic that the path is being published to
        self.publisher = self.create_publisher(Path, '/nav_path', 10)
        self.get_logger().info('test path node started')


    def send_path(self):
        path = self.generate_path()

        goal_msg = PathGen.Goal()
        goal_msg.path = path

        self._action_client.wait_for_server()
        self.publisher.publish(path)
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        return self._send_goal_future
    
    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.success))
        self.done = True
        # self.destroy_node()
        # rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback.percent_complete
        self.get_logger().info(f'recieving feedback: {feedback:2f}')
        if feedback > 0.5:
            self.request_cancel()
        

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



    def convert_to_nav_msg(self, path: List[Tuple]):
        new_path = []
        path_message = Path()

        for position in path:
            pose_message = PoseStamped()
            pose_message.header.stamp = self.get_clock().now().to_msg()
            pose_message.header.frame_id = self.frame
            pose_message.pose.position.x = float(position[0])
            pose_message.pose.position.y = float(position[1])
            new_path.append(pose_message)
        
        path_message.header.stamp = self.get_clock().now().to_msg()
        path_message.header.frame_id = self.frame
        path_message.poses = new_path

        return path_message

    def generate_path(self):
        path_array = [(0.0, 0.0), 
                      (0.03,0.0),
                      (0.06,0.0),
                      (0.09,0.03),
                      (0.12,0.06),
                      (0.15,0.06),
                      (0.18,0.06),
                      (0.21,0.06),
                      (0.24,0.06),
                      (0.27,0.03),
                      (0.30,0.0),
                      (0.33,0.0),
                      (0.36,0.0),
                      (0.36,-0.03),
                      (0.36,-0.06),
                      (0.36,-0.09),
                      (0.36,-0.12),
                      (0.36,-0.15),
                      (0.36,-0.18),
                      (0.36,-0.21),
                      (0.36,-0.24),
                      (0.36,-0.27),
                      (0.36,-0.30)]
        path_of_poses = self.convert_to_nav_msg(path_array)
        self.get_logger().info('published path!')
        return path_of_poses

def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator(0,0,0,'odom')
    node.done = False
    node.send_path()

    try:
        # while rclpy.ok() and not node.done:
        #     rclpy.spin_once(node, timeout_sec=0.1)
        # rclpy.spin_until_future_complete(node, future)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pass
        node.destroy_node()
        rclpy.shutdown() 
   
if __name__ == '__main__':
    main()
