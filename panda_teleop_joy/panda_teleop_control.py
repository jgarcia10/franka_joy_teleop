import numpy as np
import copy

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import Joy
import tf2_ros

from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive

# Helper functions
def quat2rpy(quaternion, degrees=False):
    """Convert quaternion to roll, pitch, yaw."""
    from tf_transformations import euler_from_quaternion
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
    """Convert roll, pitch, yaw to quaternion."""
    from tf_transformations import quaternion_from_euler
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
            ]
        )

        # Retrieve parameters
        self.base_frame = self.get_parameter('base_frame').value
        self.end_effector_frame = self.get_parameter('end_effector_frame').value
        self.group_name = self.get_parameter('group_name').value

        self.get_logger().debug(f"Parameters retrieved: base_frame={self.base_frame}, "
                                f"end_effector_frame={self.end_effector_frame}, group_name={self.group_name}")

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().debug("TF buffer and listener initialized.")

        # Flag to indicate if the current pose has been received
        self.current_pose_received = False

        # Flag to indicate if a goal is in progress
        self.goal_in_progress = False

        # Initialize the target pose (will be set once we receive the current pose)
        self._end_effector_target = PoseStamped()
        self._end_effector_target.header.frame_id = self.base_frame

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
        self.get_logger().info('Joystick subscriber created.')

        # Initialize MoveGroup action client
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info('MoveGroup action client initialized.')

        # Translation and rotation limits
        self._translation_limits = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]]  # x, y, z in meters
        self._rotation_limits = [[-180., 180.], [-180., 180.], [-180., 180.]]    # roll, pitch, yaw in degrees

        self.get_logger().debug(f"Translation limits: {self._translation_limits}")
        self.get_logger().debug(f"Rotation limits: {self._rotation_limits}")

        # Initialize the target pose origin (will be set once current pose is received)
        self._end_effector_target_origin = PoseStamped()

        # Create a timer for processing inputs and attempting to get the current pose
        self.timer = self.create_timer(0.1, self.timer_callback)  # Runs at 10 Hz
        self.get_logger().debug("Timer for processing inputs created (10 Hz).")

    def get_current_end_effector_pose(self):
        """Retrieve the current pose of the end effector using TF."""
        try:
            target_frame = self.base_frame
            source_frame = self.end_effector_frame
            now = rclpy.time.Time()
            timeout = rclpy.duration.Duration(seconds=1.0)

            self.get_logger().debug(f"Attempting to lookup transform from {source_frame} to {target_frame}.")

            if self.tf_buffer.can_transform(target_frame, source_frame, now, timeout):
                transform = self.tf_buffer.lookup_transform(target_frame, source_frame, now)
                self._end_effector_target.header.stamp = transform.header.stamp
                self._end_effector_target.pose.position.x = transform.transform.translation.x
                self._end_effector_target.pose.position.y = transform.transform.translation.y
                self._end_effector_target.pose.position.z = transform.transform.translation.z
                self._end_effector_target.pose.orientation = transform.transform.rotation
                self.current_pose_received = True
                self.get_logger().info('Current end-effector pose obtained from TF.')
                self.get_logger().debug(f"Pose: {self._end_effector_target.pose}")
            else:
                self.get_logger().warn(f'Transform from {source_frame} to {target_frame} not yet available.')
                self.current_pose_received = False
        except Exception as e:
            self.get_logger().error(f'Failed to get current end-effector pose: {e}')
            frames = self.tf_buffer.all_frames_as_string()
            self.get_logger().debug(f'Available frames: {frames}')
            self.current_pose_received = False

    def callback_joy(self, joy_msg):
        """Callback function to handle joystick messages."""
        self.joy_axes = joy_msg.axes
        self.joy_buttons = joy_msg.buttons
        self.get_logger().debug(f"Joystick message received: axes={self.joy_axes}, buttons={self.joy_buttons}")

    def timer_callback(self):
        """Timer callback to process joystick inputs and send goals."""
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
        """Process joystick inputs to update the target pose."""
        if not self.joy_axes:
            self.get_logger().debug('No joystick axes data to process.')
            return  # No joystick data yet

        # Define dead zone to prevent drift
        dead_zone = 0.1

        # Axes mapping with dead zones (adjust indices if necessary)
        left_stick_horizontal = self.joy_axes[0] if abs(self.joy_axes[0]) > dead_zone else 0.0  # Left/right
        left_stick_vertical = self.joy_axes[1] if abs(self.joy_axes[1]) > dead_zone else 0.0    # Forward/backward
        right_stick_horizontal = self.joy_axes[3] if abs(self.joy_axes[3]) > dead_zone else 0.0 # Yaw rotation
        right_stick_vertical = self.joy_axes[4] if abs(self.joy_axes[4]) > dead_zone else 0.0   # Up/down

        self.get_logger().debug(f"Processed joystick inputs after dead zone: "
                                f"left_stick_horizontal={left_stick_horizontal}, "
                                f"left_stick_vertical={left_stick_vertical}, "
                                f"right_stick_horizontal={right_stick_horizontal}, "
                                f"right_stick_vertical={right_stick_vertical}")

        # Buttons mapping (adjust indices if necessary)
        # Removed A button since it's for gripper control
        if len(self.joy_buttons) > 1:
            b_button = self.joy_buttons[1]  # B button to return to home position
            self.get_logger().debug(f"B button state: {b_button}")
        else:
            b_button = 0
            self.get_logger().warn("Joystick message does not have a B button index.")
        
        # Retrieve scaling factors (could be parameters)
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

        # Ensure positions are within limits
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

        self.get_logger().debug(f"Updated target pose: Position("
                                f"{self._end_effector_target.pose.position.x:.3f}, "
                                f"{self._end_effector_target.pose.position.y:.3f}, "
                                f"{self._end_effector_target.pose.position.z:.3f}), "
                                f"Orientation("
                                f"{self._end_effector_target.pose.orientation.x:.3f}, "
                                f"{self._end_effector_target.pose.orientation.y:.3f}, "
                                f"{self._end_effector_target.pose.orientation.z:.3f}, "
                                f"{self._end_effector_target.pose.orientation.w:.3f})")

        # Handle buttons
        if b_button:
            self.get_logger().info('B button pressed: Returning to home position.')
            self._home()
            return  # Don't send goal during homing

        # Check if a goal is already in progress
        if self.goal_in_progress:
            self.get_logger().debug('Goal already in progress. Skipping sending a new goal.')
            return

        # Send the target pose to MoveIt
        self.send_goal_to_moveit()

    def send_goal_to_moveit(self):
        """Send the updated target pose to MoveIt for motion planning."""
        # Set the flag indicating a goal is in progress
        self.goal_in_progress = True
        self.get_logger().debug("Goal in progress flag set to True.")

        # Retrieve parameters
        group_name = self.group_name  # 'panda_arm'

        # Create the MoveGroup goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = group_name

        # Populate the MotionPlanRequest
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = 5.0  # seconds
        goal_msg.request.workspace_parameters.header.frame_id = self.base_frame

        # Define goal constraints
        constraints = self.create_position_orientation_constraints()
        goal_msg.request.goal_constraints.append(constraints)

        # Set velocity and acceleration scaling factors
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        # Optionally, set the planner ID (e.g., 'RRTConnectkConfigDefault')
        # goal_msg.request.motion_plan_request.planner_id = 'RRTConnectkConfigDefault'

        # Debug: Log the target pose
        self.get_logger().debug(f"Target pose being sent: Position("
                                f"{self._end_effector_target.pose.position.x:.3f}, "
                                f"{self._end_effector_target.pose.position.y:.3f}, "
                                f"{self._end_effector_target.pose.position.z:.3f}), "
                                f"Orientation("
                                f"{self._end_effector_target.pose.orientation.x:.3f}, "
                                f"{self._end_effector_target.pose.orientation.y:.3f}, "
                                f"{self._end_effector_target.pose.orientation.z:.3f}, "
                                f"{self._end_effector_target.pose.orientation.w:.3f})")

        # Send the goal
        self.get_logger().info('Sending goal to MoveGroup...')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def create_position_orientation_constraints(self):
        """Create position and orientation constraints based on the target pose."""
        constraints = Constraints()

        # Position Constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.base_frame
        position_constraint.link_name = self.end_effector_frame
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Define the bounding volume for position constraint
        bounding_volume = SolidPrimitive()
        bounding_volume.type = SolidPrimitive.BOX
        bounding_volume.dimensions = [0.05, 0.05, 0.05]  # 5cm cube (adjust as needed)

        position_constraint.constraint_region.primitives.append(bounding_volume)
        position_constraint.constraint_region.primitive_poses.append(copy.deepcopy(self._end_effector_target.pose))
        position_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)

        # Orientation Constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.base_frame
        orientation_constraint.link_name = self.end_effector_frame
        orientation_constraint.orientation = self._end_effector_target.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.2  # Adjust as needed
        orientation_constraint.absolute_y_axis_tolerance = 0.2
        orientation_constraint.absolute_z_axis_tolerance = 0.2
        orientation_constraint.weight = 1.0

        constraints.orientation_constraints.append(orientation_constraint)

        self.get_logger().debug(f"Constraints created: {constraints}")
        return constraints

    def goal_response_callback(self, future):
        """Handle the response from the MoveGroup action server."""
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"MoveGroup goal response failed: {e}")
            self.goal_in_progress = False
            return

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by MoveGroup action server.')
            self.goal_in_progress = False
            return

        self.get_logger().info('Goal accepted by MoveGroup action server.')
        self.get_logger().debug('Waiting for goal result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the result from the MoveGroup action server."""
        try:
            result = future.result().result
            error_code_val = result.error_code.val

            # Reference MoveIt error codes
            # SUCCESS = 1
            # FAILURE = 99999
            # And others as per MoveIt documentation

            if error_code_val == 1:  # SUCCESS
                self.get_logger().info('MoveGroup action succeeded!')
            else:
                self.get_logger().error(f'MoveGroup action failed with error code: {error_code_val}')
        except Exception as e:
            self.get_logger().error(f'Error in getting MoveGroup action result: {e}')
        finally:
            # Reset the goal_in_progress flag regardless of success or failure
            self.goal_in_progress = False
            self.get_logger().debug('Goal in progress flag reset to False.')

    def _home(self):
        """Return the end effector to the initial pose."""
        self.get_logger().info('Returning to home position.')
        # Set the target pose to the home position (initial position)
        self._end_effector_target = copy.deepcopy(self._end_effector_target_origin)
        self._end_effector_target.header.stamp = self.get_clock().now().to_msg()
        # Log the home pose
        self.get_logger().debug(f"Home pose: Position("
                                f"{self._end_effector_target.pose.position.x:.3f}, "
                                f"{self._end_effector_target.pose.position.y:.3f}, "
                                f"{self._end_effector_target.pose.position.z:.3f}), "
                                f"Orientation("
                                f"{self._end_effector_target.pose.orientation.x:.3f}, "
                                f"{self._end_effector_target.pose.orientation.y:.3f}, "
                                f"{self._end_effector_target.pose.orientation.z:.3f}, "
                                f"{self._end_effector_target.pose.orientation.w:.3f})")
        # Send the target pose to MoveIt
        self.send_goal_to_moveit()

    def destroy_node(self):
        """Clean up before shutting down the node."""
        self.get_logger().info('Shutting down PandaTeleop node.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PandaTeleop()
    node.get_logger().info('PandaTeleop node has been created and is spinning.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
