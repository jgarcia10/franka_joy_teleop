import sys
import numpy as np
import copy

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from tf_transformations import euler_from_quaternion, quaternion_from_euler

import moveit_commander

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
        self.get_logger().info('Initializing PandaTeleop node.')

        # Initialize moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize MoveGroupCommander for the arm
        self.move_group = moveit_commander.MoveGroupCommander('panda_arm')

        # Optional: Initialize MoveGroupCommander for the gripper if needed
        # self.gripper_group = moveit_commander.MoveGroupCommander('panda_hand')

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', 'panda_link0'),          # Base frame of the robot
                ('end_effector_frame', 'panda_hand'),   # End-effector frame
                ('group_name', 'panda_arm'),            # Planning group name
                ('end_effector_name', 'panda_hand'),    # End-effector name
            ]
        )

        # Flag to indicate if the current pose has been received
        self.current_pose_received = False

        # Initialize the target pose (will be set once we receive the current pose)
        self._end_effector_target: PoseStamped = PoseStamped()
        self._end_effector_target.header.frame_id = self.get_parameter('base_frame').get_parameter_value().string_value

        # Initialize joystick variables
        self.joy_axes = []
        self.joy_buttons = []

        # Create joystick subscriber
        self._joy_subscriber = self.create_subscription(
            Joy,
            'joy',
            self.callback_joy,
            10
        )
        self.get_logger().info('Created joystick subscriber.')

        # Create a service client for actuating the gripper
        self._actuate_gripper_client = self.create_client(Empty, 'actuate_gripper')

        # Translation and rotation limits
        self._translation_limits = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]]  # x, y, z in meters
        self._rotation_limits = [[-180., 180.], [-180., 180.], [-180., 180.]]    # roll, pitch, yaw in degrees

        # Initialize the target pose origin (will be set once current pose is received)
        self._end_effector_target_origin = PoseStamped()

        # Create a timer for processing inputs and attempting to get the current pose
        self.timer = self.create_timer(0.1, self.timer_callback)  # Runs at 10 Hz

    def get_current_end_effector_pose(self):
        # Since moveit_commander already knows the current pose, we can retrieve it directly
        try:
            current_pose = self.move_group.get_current_pose().pose
            self._end_effector_target.pose = current_pose
            self.current_pose_received = True
            self.get_logger().info('Current end-effector pose obtained from MoveGroup.')
        except Exception as e:
            self.get_logger().error(f'Failed to get current end-effector pose: {e}')
            self.current_pose_received = False

    def callback_joy(self, joy_msg):
        self.joy_axes = joy_msg.axes
        self.joy_buttons = joy_msg.buttons
        self.get_logger().debug(f"Joystick axes: {self.joy_axes}")
        self.get_logger().debug(f"Joystick buttons: {self.joy_buttons}")

    def timer_callback(self):
        if not self.current_pose_received:
            self.get_current_end_effector_pose()
            if not self.current_pose_received:
                self.get_logger().info('Waiting for current end-effector pose...')
                return
            else:
                # Save the current pose as origin for teleoperation
                self._end_effector_target_origin = copy.deepcopy(self._end_effector_target)
                self.get_logger().info('Teleoperation ready.')
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
        euler_target = quat2rpy(self._end_effector_target.pose.orientation, degrees=True)

        # Update position
        self._end_effector_target.pose.position.x += delta_x
        self._end_effector_target.pose.position.y += delta_y
        self._end_effector_target.pose.position.z += delta_z

        # Update orientation
        euler_target[2] += delta_yaw  # Yaw

        # Ensure positions and orientations are within limits
        self._end_effector_target.pose.position.x = np.clip(
            self._end_effector_target.pose.position.x,
            self._translation_limits[0][0],
            self._translation_limits[0][1]
        )
        self._end_effector_target.pose.position.y = np.clip(
            self._end_effector_target.pose.position.y,
            self._translation_limits[1][0],
            self._translation_limits[1][1]
        )
        self._end_effector_target.pose.position.z = np.clip(
            self._end_effector_target.pose.position.z,
            self._translation_limits[2][0],
            self._translation_limits[2][1]
        )

        # Ensure orientations are within limits
        euler_target[0] = np.clip(
            euler_target[0],
            self._rotation_limits[0][0],
            self._rotation_limits[0][1]
        )
        euler_target[1] = np.clip(
            euler_target[1],
            self._rotation_limits[1][0],
            self._rotation_limits[1][1]
        )
        euler_target[2] = np.clip(
            euler_target[2],
            self._rotation_limits[2][0],
            self._rotation_limits[2][1]
        )

        # Convert back to quaternion
        quat = rpy2quat(euler_target, input_in_degrees=True)
        self._end_effector_target.pose.orientation = quat

        # Handle buttons
        if a_button:
            self.actuate_gripper()

        if b_button:
            self._home()
            return  # Don't send goal during homing

        # Send the target pose to MoveIt
        self.send_goal_to_moveit()

    def send_goal_to_moveit(self):
        # Set the target pose
        target_pose = Pose()
        target_pose.position.x = self._end_effector_target.pose.position.x
        target_pose.position.y = self._end_effector_target.pose.position.y
        target_pose.position.z = self._end_effector_target.pose.position.z
        target_pose.orientation = self._end_effector_target.pose.orientation

        # Set the target pose for the MoveGroup
        self.move_group.set_pose_target(target_pose)

        # Plan and execute the motion
        success = self.move_group.go(wait=True)

        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()

        # Clear targets after planning
        self.move_group.clear_pose_targets()

        if success:
            self.get_logger().info('MoveGroup action succeeded!')
        else:
            self.get_logger().error('MoveGroup action failed!')

    def actuate_gripper(self):
        # Call the service to actuate the gripper
        if not self._actuate_gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Gripper service not available')
            return

        request = Empty.Request()
        future = self._actuate_gripper_client.call_async(request)
        future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info('Gripper actuated successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to actuate gripper: {e}')

    def _home(self):
        # Set the target pose to the home position (initial position)
        self._end_effector_target = copy.deepcopy(self._end_effector_target_origin)
        # Send the target pose to MoveIt
        self.send_goal_to_moveit()

    def destroy_node(self):
        # Shut down moveit_commander
        moveit_commander.roscpp_shutdown()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PandaTeleop()
    node.get_logger().info('PandaTeleop node has been created.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
