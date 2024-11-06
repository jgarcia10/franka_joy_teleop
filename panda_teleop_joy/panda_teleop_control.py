import numpy as np
import copy

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.client import Client
from rclpy.qos import qos_profile_system_default

from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Quaternion
from tf_transformations import euler_from_quaternion, quaternion_from_euler

# Helper functions
def quat2rpy(quaternion, degrees=False):
    roll, pitch, yaw = euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w
    ])
    if degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
    return [roll, pitch, yaw]

def rpy2quat(euler, input_in_degrees=False):
    if input_in_degrees:
        euler = np.radians(euler)
    q = quaternion_from_euler(*euler)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class PandaTeleop(Node):

    def __init__(self):
        super().__init__('panda_teleop_control')
        self.get_logger().info('Initializing Panda Teleop Node')

        # Initialize parameters and variables
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', 'panda_link0'),
                ('end_effector_frame', 'panda_hand'),
                ('end_effector_target_topic', 'end_effector_target_pose'),
                ('end_effector_pose_topic', '/end_effector_pose')
            ]
        )

        # Create end effector target publisher
        self._end_effector_target_publisher: Publisher = self.create_publisher(
            Odometry,
            self.get_parameter('end_effector_target_topic').get_parameter_value().string_value,
            qos_profile_system_default
        )
        self.get_logger().info('End effector target publisher created')
        # Create end effector pose subscriber
        self._end_effector_pose_subscriber: Subscription = self.create_subscription(
            Odometry,
            self.get_parameter('end_effector_pose_topic').get_parameter_value().string_value,
            self.callback_end_effector_odom,
            10
        )
        self.get_logger().info('End effector pose subscriber created')
        # Create joystick subscriber
        self._joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.callback_joy,
            qos_profile_system_default
        )
        self.get_logger().info('Joystick subscriber created')
        # Create a service client for actuating the gripper
        self._actuate_gripper_client: Client = self.create_client(Empty, 'actuate_gripper')

        # Initialize variables
        self.joy_axes = []
        self.joy_buttons = []

        # Initialize end effector target and pose
        self._end_effector_target_origin: Odometry = Odometry()
        self._end_effector_target_origin.pose.pose.position.x = 0.3070
        self._end_effector_target_origin.pose.pose.position.y = 0.0
        self._end_effector_target_origin.pose.pose.position.z = 0.4872
        self._end_effector_target_origin.pose.pose.orientation = Quaternion(
            x=-0.00014, y=0.7071, z=0.00014, w=0.7071)
        self._end_effector_target_origin.header.frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
        self._end_effector_target_origin.child_frame_id = self.get_parameter('end_effector_frame').get_parameter_value().string_value
        self._end_effector_target_origin.header.stamp = self.get_clock().now().to_msg()

        self._end_effector_target: Odometry = copy.deepcopy(self._end_effector_target_origin)
        self._end_effector_pose: Odometry = copy.deepcopy(self._end_effector_target)

        # Publish the initial end effector target
        self._end_effector_target_publisher.publish(self._end_effector_target)

        # Translation and rotation limits
        self._translation_limits = [[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]]  # x, y, z in meters
        self._rotation_limits = [[-90., 90.], [-90., 90.], [-90., 90.]]    # roll, pitch, yaw in degrees

        # Create timer for processing inputs
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

    def callback_end_effector_odom(self, odom: Odometry):
        self._end_effector_pose = odom

    def callback_joy(self, joy_msg):
        self.joy_axes = joy_msg.axes
        self.joy_buttons = joy_msg.buttons
        self.get_logger().info(f"Joystick axes: {self.joy_axes}")
        self.get_logger().info(f"Joystick buttons: {self.joy_buttons}")

    def timer_callback(self):
        self.process_joy_input()

    def process_joy_input(self):
        if not self.joy_axes:
            return  # No joystick data yet

        # Axes mapping (adjust indices if necessary)
        left_stick_horizontal = self.joy_axes[0]  # Left stick horizontal (left/right)
        left_stick_vertical = self.joy_axes[1]    # Left stick vertical (forward/backward)
        right_stick_horizontal = self.joy_axes[3] # Right stick horizontal (yaw rotation)
        right_stick_vertical = self.joy_axes[4]   # Right stick vertical (up/down)

        # Buttons mapping (adjust indices if necessary)
        a_button = self.joy_buttons[0]  # A button to open/close gripper
        b_button = self.joy_buttons[1]  # B button to return to home position

        # Scaling factors
        translation_speed = 0.01  # meters per update
        rotation_speed = 1.0      # degrees per update

        # Update end effector position
        delta_x = left_stick_vertical * translation_speed  # Forward/Backward
        delta_y = left_stick_horizontal * translation_speed  # Left/Right
        delta_z = right_stick_vertical * translation_speed  # Up/Down

        # Update orientation (Yaw rotation)
        delta_yaw = right_stick_horizontal * rotation_speed

        # Get current orientation in Euler angles
        euler_target = quat2rpy(self._end_effector_target.pose.pose.orientation, degrees=True)

        # Update position
        self._end_effector_target.pose.pose.position.x += delta_x
        self._end_effector_target.pose.pose.position.y += delta_y
        self._end_effector_target.pose.pose.position.z += delta_z

        # Update orientation
        euler_target[2] += delta_yaw  # Yaw

        # Ensure positions and orientations are within limits
        self._end_effector_target.pose.pose.position.x = np.clip(
            self._end_effector_target.pose.pose.position.x,
            self._translation_limits[0][0],
            self._translation_limits[0][1]
        )
        self._end_effector_target.pose.pose.position.y = np.clip(
            self._end_effector_target.pose.pose.position.y,
            self._translation_limits[1][0],
            self._translation_limits[1][1]
        )
        self._end_effector_target.pose.pose.position.z = np.clip(
            self._end_effector_target.pose.pose.position.z,
            self._translation_limits[2][0],
            self._translation_limits[2][1]
        )

        # Convert back to quaternion
        quat = rpy2quat(euler_target, input_in_degrees=True)
        self._end_effector_target.pose.pose.orientation = quat

        # Handle buttons
        if a_button:
            # Open/Close gripper
            self.actuate_gripper()

        if b_button:
            # Return to home position
            self._home()

        # Publish the updated target
        self._publish()

    def actuate_gripper(self):
        # Call the service to actuate the gripper
        if not self._actuate_gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Gripper service not available')
            return

        future = self._actuate_gripper_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Gripper actuated successfully')
        else:
            self.get_logger().error('Failed to actuate gripper')

    def _home(self):
        self._end_effector_target = copy.deepcopy(self._end_effector_target_origin)
        self._end_effector_target.header.stamp = self.get_clock().now().to_msg()
        self._publish()

    def _publish(self):
        self._end_effector_target.header.stamp = self.get_clock().now().to_msg()
        self._end_effector_target_publisher.publish(self._end_effector_target)

def main(args=None):
    rclpy.init(args=args)
    node = PandaTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

