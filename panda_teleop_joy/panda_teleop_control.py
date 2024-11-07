import numpy as np
import copy

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs

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

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        # Initialize MoveGroup action client
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info('MoveGroup action client initialized.')

        # Translation and rotation limits
        self._translation_limits = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]]  # x, y, z in meters
        self._rotation_limits = [[-180., 180.], [-180., 180.], [-180., 180.]]    # roll, pitch, yaw in degrees

        # Initialize the target pose origin (will be set once current pose is received)
        self._end_effector_target_origin = PoseStamped()

        # Create a timer for processing inputs and attempting to get the current pose
        self.timer = self.create_timer(0.1, self.timer_callback)  # Runs at 10 Hz

    def get_current_end_effector_pose(self):
        try:
            target_frame = self.get_parameter('base_frame').get_parameter_value().string_value
            source_frame = self.get_parameter('end_effector_frame').get_parameter_value().string_value
            now = rclpy.time.Time()
            timeout = rclpy.duration.Duration(seconds=1.0)

            if self.tf_buffer.can_transform(target_frame, source_frame, now, timeout):
                transform = self.tf_buffer.lookup_transform(target_frame, source_frame, now)
                self._end_effector_target.header.stamp = transform.header.stamp
                self._end_effector_target.pose.position.x = transform.transform.translation.x
                self._end_effector_target.pose.position.y = transform.transform.translation.y
                self._end_effector_target.pose.position.z = transform.transform.translation.z
                self._end_effector_target.pose.orientation = transform.transform.rotation
                self.current_pose_received = True
                self.get_logger().info('Current end-effector pose obtained from TF.')
            else:
                self.get_logger().warn(f'Transform from {source_frame} to {target_frame} not yet available.')
                self.current_pose_received = False
        except Exception as e:
            self.get_logger().error(f'Failed to get current end-effector pose: {e}')
            frames = self.tf_buffer.all_frames_as_string()
            self.get_logger().debug(f'Available frames: {frames}')
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
        # Ensure the action server is available
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available. Ensure that MoveIt is running.')
            return

        # Retrieve parameters
        group_name = self.get_parameter('group_name').value            # 'panda_arm'
        end_effector_name = self.get_parameter('end_effector_name').value  # 'panda_hand'

        # Create the MoveGroup goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = group_name
        goal_msg.request.end_effector_name = end_effector_name

        # Populate the MotionPlanRequest
        goal_msg.request.motion_plan_request.group_name = group_name
        goal_msg.request.motion_plan_request.num_planning_attempts = 5
        goal_msg.request.motion_plan_request.allowed_planning_time = 5.0  # seconds
        goal_msg.request.motion_plan_request.workspace_parameters.header.frame_id = self._end_effector_target.header.frame_id

        # Define goal constraints
        constraints = self.create_position_orientation_constraints()
        goal_msg.request.motion_plan_request.goal_constraints.append(constraints)

        # Optionally, define path constraints if needed
        # goal_msg.request.motion_plan_request.path_constraints = your_path_constraints

        # Set velocity and acceleration scaling factors
        goal_msg.request.motion_plan_request.max_velocity_scaling_factor = 0.1
        goal_msg.request.motion_plan_request.max_acceleration_scaling_factor = 0.1

        # Optionally, set the planner ID (e.g., 'RRTConnectkConfigDefault')
        # goal_msg.request.motion_plan_request.planner_id = 'RRTConnectkConfigDefault'

        # Send the goal
        self.get_logger().info('Sending goal to MoveGroup...')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def create_position_orientation_constraints(self):
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive

        constraints = Constraints()

        # Position Constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self._end_effector_target.header.frame_id
        position_constraint.link_name = self.get_parameter('end_effector_frame').value
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Define the bounding volume for position constraint
        bounding_volume = SolidPrimitive()
        bounding_volume.type = SolidPrimitive.BOX
        bounding_volume.dimensions = [0.01, 0.01, 0.01]  # 1cm cube

        position_constraint.constraint_region.primitives.append(bounding_volume)
        position_constraint.constraint_region.primitive_poses.append(self._end_effector_target.pose)
        position_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)

        # Orientation Constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self._end_effector_target.header.frame_id
        orientation_constraint.link_name = self.get_parameter('end_effector_frame').value
        orientation_constraint.orientation = self._end_effector_target.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        constraints.orientation_constraints.append(orientation_constraint)

        return constraints

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by MoveGroup action server')
            return

        self.get_logger().info('Goal accepted by MoveGroup action server')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info('MoveGroup action succeeded!')
        else:
            self.get_logger().error(f'MoveGroup action failed with error code: {result.error_code.val}')

    def actuate_gripper(self):
        # Call the service to actuate the gripper
        if not self._actuate_gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Gripper service not available')
            return

        future = self._actuate_gripper_client.call_async(Empty.Request())
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
        self._end_effector_target.header.stamp = self.get_clock().now().to_msg()
        # Send the target pose to MoveIt
        self.send_goal_to_moveit()

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
